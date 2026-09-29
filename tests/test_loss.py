"""The Doppler weight w_D is an exact reparameterization of the plain L2 (paper Sec. 4.4)."""

import torch

from dyrad.loss import reconstruction_loss


def _pair(D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.rand(2, D, 12, 20, generator=g, dtype=torch.float64)
    gt = torch.rand(2, D, 12, 20, generator=g, dtype=torch.float64)
    return pred, gt


def test_w1_is_plain_l2():
    pred, gt = _pair()
    plain = (pred - gt).pow(2).mean()
    assert torch.allclose(reconstruction_loss(pred, gt, 1.0), plain)
    # the split form at w_D=1 reproduces the plain L2 too (the cross term vanishes)
    pm, gm = pred.mean(1, keepdim=True), gt.mean(1, keepdim=True)
    split = ((pm - gm).pow(2) + ((pred - pm) - (gt - gm)).pow(2).mean(1, keepdim=True)).mean()
    assert torch.allclose(split, plain, atol=1e-12)


def test_w0_is_doppler_averaged_ra_loss():
    pred, gt = _pair()
    ra = (pred.mean(1) - gt.mean(1)).pow(2).mean()
    assert torch.allclose(reconstruction_loss(pred, gt, 0.0), ra, atol=1e-12)


def test_monotone_in_wd():
    pred, gt = _pair()
    vals = [reconstruction_loss(pred, gt, w).item() for w in (0.0, 0.25, 0.5, 1.0, 2.0)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))


def test_single_doppler_bin_ignores_wd():
    pred, gt = _pair(D=1)
    base = (pred - gt).pow(2).mean()
    for w in (0.0, 0.5, 7.0):
        assert torch.allclose(reconstruction_loss(pred, gt, w), base)
