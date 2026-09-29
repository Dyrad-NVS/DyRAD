"""The reconstruction loss L_rec (paper Sec. 3.3) with the Doppler-supervision weight of Sec. 4.4."""

from __future__ import annotations

from torch import Tensor


def reconstruction_loss(pred: Tensor, gt: Tensor, doppler_axis_weight: float = 1.0) -> Tensor:
    """Mean squared error over every RAD bin, L_rec = mean((Y_hat - Y)^2).

    `doppler_axis_weight` (w_D) scales the Doppler-shape part of the error only. Each (R, A)
    cell's Doppler profile splits exactly into its mean over D and a zero-mean shape, so the
    cross term vanishes and

        mean_D (dc + shape_d)^2  ==  dc^2 + mean_D shape_d^2.

    w_D = 1 is therefore the plain L2 and w_D = 0 the loss on the Doppler-averaged RA map
    (the "w/o Doppler" ablation). A length-1 Doppler axis has no shape component.
    Tensors are [B, D, R, A].
    """
    if doppler_axis_weight == 1.0 or pred.shape[1] == 1:
        return (pred - gt).pow(2).mean()
    pm = pred.mean(dim=1, keepdim=True)
    gm = gt.mean(dim=1, keepdim=True)
    dc = pm - gm
    shape = (pred - pm) - (gt - gm)
    return (dc.pow(2) + doppler_axis_weight * shape.pow(2).mean(dim=1, keepdim=True)).mean()
