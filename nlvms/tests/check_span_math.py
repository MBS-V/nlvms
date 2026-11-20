"""Pure-Python check of the IoU/GIoU formulas. No dependencies."""


def iou_np(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(0.0, a[1] - a[0]) + max(0.0, b[1] - b[0]) - inter
    return inter / max(union, 1e-6)


def giou_np(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(0.0, a[1] - a[0]) + max(0.0, b[1] - b[0]) - inter
    enclose = max(a[1], b[1]) - min(a[0], b[0])
    return inter / max(union, 1e-6) - (enclose - union) / enclose


cases = [
    ((42, 58), (42, 58), 1.0, "perfect match"),
    ((38, 62), (42, 58), 16 / 24, "overlap 16, union 24"),
    ((0, 10), (20, 30), 0.0, "disjoint"),
    ((0, 10), (5, 15), 5 / 15, "half overlap"),
]
for pred, gt, expected, label in cases:
    got = iou_np(pred, gt)
    assert abs(got - expected) < 1e-6, label
    print(f"[OK] {label}: {got:.4f}")

assert giou_np((0, 10), (20, 30)) < 0
assert giou_np((0, 10), (5, 15)) <= iou_np((0, 10), (5, 15)) + 1e-9
print("all span math checks passed")
