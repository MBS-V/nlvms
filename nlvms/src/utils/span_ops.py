"""
Conversions and overlap measures for 1D temporal spans.

Spans appear in two forms:
  (center, width)  - what the regression head predicts
  (start, end)     - what losses and metrics compare against ground truth
"""

import torch


def cw_to_se(spans):
    center, width = spans.unbind(-1)
    return torch.stack([center - 0.5 * width, center + 0.5 * width], dim=-1)


def se_to_cw(spans):
    start, end = spans.unbind(-1)
    return torch.stack([0.5 * (start + end), end - start], dim=-1)


def _inter_union(spans_a, spans_b):
    start_a, end_a = spans_a.unbind(-1)
    start_b, end_b = spans_b.unbind(-1)
    inter_start = torch.max(start_a[:, None], start_b[None, :])
    inter_end = torch.min(end_a[:, None], end_b[None, :])
    inter = (inter_end - inter_start).clamp(min=0)
    len_a = (end_a - start_a).clamp(min=0)
    len_b = (end_b - start_b).clamp(min=0)
    union = len_a[:, None] + len_b[None, :] - inter
    return inter, union


def pairwise_iou(spans_a, spans_b):
    """(N,2) start/end, (M,2) start/end -> (N,M) IoU."""
    inter, union = _inter_union(spans_a, spans_b)
    return inter / union.clamp(min=1e-6)


def pairwise_giou(spans_a, spans_b):
    """Generalized IoU for intervals. (N,2), (M,2) -> (N,M) in [-1, 1]."""
    inter, union = _inter_union(spans_a, spans_b)
    iou = inter / union.clamp(min=1e-6)
    start_a, end_a = spans_a.unbind(-1)
    start_b, end_b = spans_b.unbind(-1)
    enclose = (torch.max(end_a[:, None], end_b[None, :])
               - torch.min(start_a[:, None], start_b[None, :])).clamp(min=1e-6)
    return iou - (enclose - union) / enclose


def elementwise_iou(spans_a, spans_b):
    """Row-by-row IoU. (N,2), (N,2) -> (N,)."""
    start_a, end_a = spans_a.unbind(-1)
    start_b, end_b = spans_b.unbind(-1)
    inter = (torch.min(end_a, end_b) - torch.max(start_a, start_b)).clamp(min=0)
    union = (end_a - start_a).clamp(min=0) + (end_b - start_b).clamp(min=0) - inter
    return inter / union.clamp(min=1e-6)
