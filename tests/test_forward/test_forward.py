import finufft
import numpy as np
import pytest
import scipy
import torch
from numpy.random import standard_normal

import pytorch_finufft

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

Ns = [10, 100, 1000, 10000]


@pytest.mark.parametrize(
    "values",
    [
        (standard_normal(size=Ns[0]) + 1j * standard_normal(size=Ns[0])),
        (standard_normal(size=Ns[1]) + 1j * standard_normal(size=Ns[1])),
        (standard_normal(size=Ns[2]) + 1j * standard_normal(size=Ns[2])),
        (standard_normal(size=Ns[3]) + 1j * standard_normal(size=Ns[3])),
    ],
)
def test_1d_t1_forward_CPU(values: np.ndarray) -> None:
    """
    Basic test cases for 1d Type 1 against Pytorch and Numpy/ Scipy

    Tests against existing (uniform/ standard) FFT implementations by setting up a uniform grid for FINUFFT.
    """
    N = len(values)
    val_tens = torch.from_numpy(values)

    print(val_tens.dtype)

    against_torch = torch.fft.fft(val_tens)
    against_scipy = torch.from_numpy(scipy.fft.fft(values))
    against_numpy = torch.from_numpy(np.fft.fft(values))

    finufft1D1_out = pytorch_finufft.functional.finufft1D1.forward(
        2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64), val_tens, N
    )

    assert against_torch.dtype == val_tens.dtype
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
    "targets",
    [
        (standard_normal(size=Ns[0]) + 1j * standard_normal(size=Ns[0])),
        (standard_normal(size=Ns[1]) + 1j * standard_normal(size=Ns[1])),
        (standard_normal(size=Ns[2]) + 1j * standard_normal(size=Ns[2])),
        (standard_normal(size=Ns[3]) + 1j * standard_normal(size=Ns[3])),
    ],
)
def test_1d_t2_forward_CPU(targets: torch.Tensor):
    """
    Test type 2 API against existing implementations
    """
    N = len(targets)
    target_tens = torch.from_numpy(targets)

    against_torch = torch.fft.fft(target_tens)
    against_scipy = scipy.fft.fft(targets)
    against_numpy = np.fft.fft(targets)

    finufft_out = pytorch_finufft.functional.finufft1D2.forward(
        2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64), target_tens
    )

    assert against_torch.dtype == target_tens.dtype
    assert (
        torch.linalg.norm(finufft_out - against_torch) / N
    ) == pytest.approx(0, abs=1e-05, rel=1e-06)
    assert (
        torch.linalg.norm(finufft_out - against_scipy) / N
    ) == pytest.approx(0, abs=1e-05, rel=1e-06)
    assert (
        torch.linalg.norm(finufft_out - against_numpy) / N
    ) == pytest.approx(0, abs=1e-05, rel=1e-06)


@pytest.mark.parametrize(
    "values",
    [
        (standard_normal(size=Ns[0]) + 1j * standard_normal(size=Ns[0])),
        (standard_normal(size=Ns[1]) + 1j * standard_normal(size=Ns[1])),
        (standard_normal(size=Ns[2]) + 1j * standard_normal(size=Ns[2])),
        (standard_normal(size=Ns[3]) + 1j * standard_normal(size=Ns[3])),
    ],
)
def test_1d_t3_forward_CPU(values: torch.Tensor):
    """
    Test type 3 API against existing implementations
    """
    N = len(values)
    val_tens = torch.from_numpy(values)

    against_torch = torch.fft.fft(val_tens)
    against_scipy = scipy.fft.fft(values)
    against_numpy = np.fft.fft(values)

    finufft_out = pytorch_finufft.functional.finufft1D3.forward(
        points=2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64),
        values=val_tens,
        # targets=2 * np.pi * torch.arange(0, 1, 1/N, dtype=torch.float64)
        # targets=val_tens,
        targets=torch.ones(N),
    )

    assert against_torch.dtype == finufft_out.dtype
    assert (
        torch.linalg.norm(finufft_out - against_torch) / N
    ) == pytest.approx(0, abs=1e-05, rel=1e-06)
    assert (
        torch.linalg.norm(finufft_out - against_scipy) / N
    ) == pytest.approx(0, abs=1e-05, rel=1e-06)
    assert (
        torch.linalg.norm(finufft_out - against_numpy) / N
    ) == pytest.approx(0, abs=1e-05, rel=1e-06)
