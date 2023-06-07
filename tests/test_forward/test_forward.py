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
        (standard_normal(size=Ns[0]) + 1j * standard_normal(size=Ns[0])),
        (standard_normal(size=Ns[1]) + 1j * standard_normal(size=Ns[1])),
        (standard_normal(size=Ns[2]) + 1j * standard_normal(size=Ns[2])),
    ],
)
def test_1d_t1_forward_CPU(c: np.ndarray) -> None:
    """
    Basic test cases for 1d Type 1 against Pytorch and FINUFFT

    Tests against existing (uniform/ standard) FFT implementations by setting up a uniform grid for FINUFFT.
    """
    N = len(c)
    ctens = torch.from_numpy(c)

    against_torch = torch.fft.fft(ctens)
    against_scipy = torch.from_numpy(scipy.fft.fft(c))
    against_numpy = torch.from_numpy(np.fft.fft(c))

    finufft_out = pytorch_finufft.functional.finufft1D1.forward(
        torch.from_numpy(2 * np.pi * np.arange(0, 1, 1 / N)), ctens, N
    )

    assert torch.linalg.norm(finufft_out - against_torch) == pytest.approx(
        0, abs=1e-06
    )
    assert torch.linalg.norm(finufft_out - against_scipy) == pytest.approx(
        0, abs=1e-06
    )
    assert torch.linalg.norm(finufft_out - against_numpy) == pytest.approx(
        0, abs=1e-06
    )


@pytest.mark.parametrize(
    ("c"),
    [
        (standard_normal(size=Ns[0])),
        (standard_normal(size=Ns[1])),
    ],
)
def test_1d_t2_forward_CPU(c: torch.Tensor):
    """
    Test type 2 API against existing implementations
    """
    N = len(c)
    ctens = torch.from_numpy(c)

    against_torch = torch.fft.fft(ctens)

    # out = pytorch_finufft.functionl.finufft1D2.forward(
    # 2 * np.pi * np.arange(0, 1, 1/N), c
    # )
