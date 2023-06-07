"""Tests for hello function."""
import finufft
import numpy as np
import pytest
import scipy
import torch
from numpy.random import standard_normal

import pytorch_finufft

Ns = [100, 1000, 10000]


@pytest.mark.parametrize(
    ("c"),
    [
        (
            torch.from_numpy(
                standard_normal(size=Ns[0]) + 1j * standard_normal(size=Ns[0])
            )
        ),
        (
            torch.from_numpy(
                standard_normal(size=Ns[1]) + 1j * standard_normal(size=Ns[1])
            )
        ),
        (
            torch.from_numpy(
                standard_normal(size=Ns[2]) + 1j * standard_normal(size=Ns[2])
            )
        ),
    ],
)
def test_1d_t1_forward_CPU(c: torch.Tensor) -> None:
    """
    Basic test cases for 1d `pytorch_finufft` bindings against the `finufft` library itself and several
    uniform cases as well.

    Tests against existing (uniform/ standard) FFT implementations by setting up a uniform grid for FINUFFT.
    """
    N = len(c)
    against_np = np.fft.fft(c.item())
    against_scipy = scipy.fft.fft(c.item())
    against_torch = torch.fft.fft(c)
    against_finufft = finufft.nufft1d1(
        2 * np.pi * np.arange(0, 1, 1 / N),
        c.item(),
        N,
        eps=1e-08,
        isign=-1,
        modeord=1,
    )

    out = pytorch_finufft.functional.finufft1D1.forward(
        2 * np.pi * np.arange(0, 1, 1 / N), c.item(), N, eps=1e-08
    )

    assert torch.linalg.norm(out - against_np) == pytest.approx(0, abs=1e-03)
    assert torch.linalg.norm(out - against_scipy) == pytest.approx(0, abs=1e-03)
    assert torch.linalg.norm(out - against_torch) == pytest.approx(0, abs=1e-03)
    assert torch.linalg.norm(out - against_finufft) == pytest.approx(
        0, abs=1e-08
    )
