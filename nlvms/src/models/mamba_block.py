"""
Selective state-space (Mamba) layer with bidirectional wrapping.

Two paths behind one interface, forward(x): (B, L, D) -> (B, L, D):
  - official CUDA kernel (mamba_ssm.Mamba), used when installed and a GPU is present
  - a plain PyTorch selective scan, used otherwise (slow, same math)

The fallback runs in fp32 even under autocast: the exp(delta*A) terms are
numerically touchy in fp16, and the T4 has no bf16.

Build the model after confirming the GPU is available, since the path is
chosen when each layer is constructed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _CudaMamba
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False


class SelectiveScanFallback(nn.Module):
    def __init__(self, dim, state_dim=32, conv_kernel=4, expand=2):
        super().__init__()
        self.state_dim = state_dim
        self.inner_dim = expand * dim

        self.in_proj = nn.Linear(dim, 2 * self.inner_dim, bias=False)
        self.conv1d = nn.Conv1d(
            self.inner_dim, self.inner_dim, kernel_size=conv_kernel,
            groups=self.inner_dim, padding=conv_kernel - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.inner_dim, self.inner_dim + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)

        A = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(self.inner_dim, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)

    def forward(self, x):
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = self._scan(x.float())
        return out.to(in_dtype)

    def _scan(self, x):
        batch, length, _ = x.shape
        x_in, z = self.in_proj(x).chunk(2, dim=-1)

        x_in = self.conv1d(x_in.transpose(1, 2))[:, :, :length].transpose(1, 2)
        x_in = F.silu(x_in)

        delta, B, C = torch.split(
            self.x_proj(x_in), [self.inner_dim, self.state_dim, self.state_dim], dim=-1)
        delta = F.softplus(self.dt_proj(delta))            # (B, L, D)

        A = -torch.exp(self.A_log)                         # (D, N)
        delta_A = torch.exp(delta.unsqueeze(-1) * A)       # (B, L, D, N)
        delta_Bx = delta.unsqueeze(-1) * B.unsqueeze(2) * x_in.unsqueeze(-1)

        h = x_in.new_zeros(batch, self.inner_dim, self.state_dim)
        ys = []
        for t in range(length):
            h = delta_A[:, t] * h + delta_Bx[:, t]
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1)

        y = (y + x_in * self.D) * F.silu(z)
        return self.out_proj(y)


def make_mamba_layer(dim, state_dim=32):
    if MAMBA_SSM_AVAILABLE and torch.cuda.is_available():
        return _CudaMamba(d_model=dim, d_state=state_dim)
    return SelectiveScanFallback(dim, state_dim=state_dim)


class BidirectionalMamba(nn.Module):
    """
    Two independently weighted Mamba layers, one reading forward and one
    reading the reversed sequence, summed. Separate weights because
    forward-time and backward-time dynamics are different tasks.
    """

    def __init__(self, dim, state_dim=32):
        super().__init__()
        self.forward_mamba = make_mamba_layer(dim, state_dim)
        self.backward_mamba = make_mamba_layer(dim, state_dim)

    def forward(self, x, padding_mask=None):
        """
        x: (B, L, D); padding_mask: (B, L) bool, True where real.
        Padding is zeroed before scanning. In the reversed pass the padding
        becomes leading zeros, which the scan treats as near-empty input.
        """
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(x.dtype)

        fwd = self.forward_mamba(x)
        bwd = torch.flip(self.backward_mamba(torch.flip(x, dims=[1])), dims=[1])
        return fwd + bwd
