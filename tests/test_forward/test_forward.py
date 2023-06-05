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
    ("c", "N"),
    [
        ((standard_normal(size=Ns[0]) + 1j * standard_normal(size=Ns[0])), Ns[0]),
        ((standard_normal(size=Ns[1]) + 1j * standard_normal(size=Ns[1])), Ns[1]),
        ((standard_normal(size=Ns[2]) + 1j * standard_normal(size=Ns[2])), Ns[2]),
    ],
)
def test_1d_t1_forward_CPU(c: np.ndarray, N: np.int64) -> None:
    """
    Basic test cases for 1d `pytorch_finufft` bindings against the `finufft` library itself and several
    uniform cases as well.

    Tests against existing (uniform/ standard) FFT implementations by setting up a uniform grid for FINUFFT.
    """
    against_np = np.fft.fft(c)
    against_scipy = scipy.fft.fft(c)
    against_torch = torch.fft.fft(torch.from_numpy(c))
    against_finufft = finufft.nufft1d1(
        2 * np.pi * np.arange(0, 1, 1 / len(c)), c, N, eps=1e-08, isign=-1, modeord=1
    )

    out = pytorch_finufft.functional.finufft1D1.forward(
        2 * np.pi * np.arange(0, 1, 1 / len(c)), c, N, eps=1e-08
    )

    assert torch.linalg.norm(out - against_np) == pytest.approx(0, abs=1e-03)
    assert torch.linalg.norm(out - against_scipy) == pytest.approx(0, abs=1e-03)
    assert torch.linalg.norm(out - against_torch) == pytest.approx(0, abs=1e-03)
    assert torch.linalg.norm(out - against_finufft) == pytest.approx(0, abs=1e-08)


@pytest.mark.parametrize(("something", "N"), [("temp", "temp2")])
def test_1d_t2_forward_CPU(something, N):
    pass
