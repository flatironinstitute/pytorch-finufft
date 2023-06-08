"""Tests for hello function."""
import finufft
import numpy as np
import pytest
import scipy
import torch
from numpy.random import standard_normal

import pytorch_finufft

Ns = [10, 100, 1000]


@pytest.mark.parametrize(
    "c",
    [
        (standard_normal(size=Ns[0]) + 1j * standard_normal(size=Ns[0])),
        (standard_normal(size=Ns[1]) + 1j * standard_normal(size=Ns[1])),
        (standard_normal(size=Ns[2]) + 1j * standard_normal(size=Ns[2])),
    ],
)
def test_1d_t1_forward_CPU(c: np.ndarray) -> None:
    """
    Basic test cases for 1d Type 1 against Pytorch and Numpy/ Scipy

    Tests against existing (uniform/ standard) FFT implementations by setting up a uniform grid for FINUFFT.
    """
    N = len(c)
    ctens = torch.from_numpy(c)

    print(ctens.dtype)

    against_torch = torch.fft.fft(ctens)
    against_scipy = torch.from_numpy(scipy.fft.fft(c))
    against_numpy = torch.from_numpy(np.fft.fft(c))

    finufft1D1_out = pytorch_finufft.functional.finufft1D1.forward(
        2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64), ctens, N
    )

    assert type(against_torch) == type(ctens)
    assert (
        torch.linalg.norm(finufft1D1_out - against_torch) / N
    ) == pytest.approx(0, abs=1e-05, rel=1e-06)
    assert (
        torch.linalg.norm(finufft1D1_out - against_scipy) / N
    ) == pytest.approx(0, abs=1e-05, rel=1e-06)
    assert (
        torch.linalg.norm(finufft1D1_out - against_numpy) / N
    ) == pytest.approx(0, abs=1e-05, rel=1e-06)


@pytest.mark.parametrize(
    "c",
    [
        (standard_normal(size=Ns[0]) + 1j * standard_normal(size=Ns[0])),
        (standard_normal(size=Ns[1]) + 1j * standard_normal(size=Ns[1])),
    ],
)
def test_1d_t2_forward_CPU(c: torch.Tensor):
    """
    Test type 2 API against existing implementations
    """
    N = len(c)
    ctens = torch.from_numpy(c)

    against_torch = torch.fft.fft(ctens)
    against_scipy = scipy.fft.fft(c)
    against_numpy = np.fft.fft(c)

    finufft_out = pytorch_finufft.functional.finufft1D2.forward(
        2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64), ctens
    )

    assert type(against_torch) == type(ctens)
    assert (
        torch.linalg.norm(finufft_out - against_torch) / N
    ) == pytest.approx(0, abs=1e-05, rel=1e-06)
    assert (
        torch.linalg.norm(finufft_out - against_scipy) / N
    ) == pytest.approx(0, abs=1e-05, rel=1e-06)
    assert (
        torch.linalg.norm(finufft_out - against_numpy) / N
    ) == pytest.approx(0, abs=1e-05, rel=1e-06)
