import numpy as np
import pytest
import scipy
import torch
from numpy.random import standard_normal

import pytorch_finufft

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

# Case generation
Ns = [
    10,
    15,
    100,
    101,
    1000,
    1001,
    2500,
    3750,
    5000,
    5001,
    6250,
    7500,
    8750,
    10000,
]
cases = [np.array([1.0, 2.5, -1.0, -1.5, 1.5], dtype=np.complex128)]
for n in Ns:
    cases.append(standard_normal(n) + 1j * standard_normal(n))


# Tests
@pytest.mark.parametrize("values", cases)
def test_1d_t1_forward_CPU(values: np.ndarray) -> None:
    """
    Tests against implementations of the FFT by setting up a uniform grid over
    which to call FINUFFT through the API.
    """
    N = len(values)
    val_tens = torch.from_numpy(values)

    against_torch = torch.fft.fft(val_tens)
    against_scipy = torch.from_numpy(scipy.fft.fft(values))

    finufft1D1_out = pytorch_finufft.functional.finufft1D1.forward(
        2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64),
        val_tens,
        N,
    )

    assert against_torch.dtype == val_tens.dtype
    assert (
        torch.linalg.norm(finufft1D1_out - against_torch) / (N**2)
    ) == pytest.approx(0, abs=1e-06)
    assert (
        torch.linalg.norm(finufft1D1_out - against_scipy) / (N**2)
    ) == pytest.approx(0, abs=1e-06)


@pytest.mark.parametrize("targets", cases)
def test_1d_t2_forward_CPU(targets: np.ndarray):
    """
    Test type 2 API against existing implementations by setting
    """
    N = len(targets)
    inv_targets = scipy.fft.fft(targets)
    inv_targets_tens = torch.from_numpy(inv_targets)

    assert len(inv_targets) == N

    against_torch = torch.fft.ifft(inv_targets_tens)

    finufft_out = (
        pytorch_finufft.functional.finufft1D2.forward(
            2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64),
            inv_targets_tens,
            isign=1,
            modeord=1,
        )
        / N
    )

    assert torch.norm(finufft_out - np.array(targets)) / (
        N**2
    ) == pytest.approx(0, abs=1e-05)
    assert torch.norm(finufft_out - against_torch) / (N**2) == pytest.approx(
        0, abs=1e-05
    )


@pytest.mark.parametrize("values", cases)
def test_1d_t3_forward_CPU(values: np.ndarray):
    """
    Test type 3 API against existing implementations
    """
    N = len(values)

    points = 2 * np.pi * torch.arange(0, 1, 1 / N)
    targets = torch.arange(0, N, dtype=torch.float64)

    finufft_out = pytorch_finufft.functional.finufft1D3.forward(
        points, torch.from_numpy(values), targets
    )

    against_torch = torch.fft.fft(torch.from_numpy(values))
    against_scipy = scipy.fft.fft(values)

    assert against_torch.dtype == finufft_out.dtype
    assert (
        torch.linalg.norm(finufft_out - against_torch) / (N**2)
    ) == pytest.approx(0, abs=1e-05)
    assert (
        torch.linalg.norm(finufft_out - torch.from_numpy(against_scipy))
        / (N**2)
    ) == pytest.approx(0, abs=1e-05)
