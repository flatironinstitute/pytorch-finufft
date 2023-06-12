import numpy as np
import pytest
import scipy
import torch
from numpy.random import standard_normal

import pytorch_finufft

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

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
        torch.linalg.norm(finufft1D1_out - against_torch) / N
    ) == pytest.approx(0, abs=1e-05, rel=1e-06)
    assert (
        torch.linalg.norm(finufft1D1_out - against_scipy) / N
    ) == pytest.approx(0, abs=1e-05, rel=1e-06)


@pytest.mark.parametrize(
    "targets",
    [
        ([1.0, 2.0, 1.0, -1.0, 1.5]),
        # (standard_normal(size=Ns[0]) + 1j * standard_normal(size=Ns[0])),
        # (standard_normal(size=Ns[1]) + 1j * standard_normal(size=Ns[1])),
        # (standard_normal(size=Ns[2]) + 1j * standard_normal(size=Ns[2])),
        # (standard_normal(size=Ns[3]) + 1j * standard_normal(size=Ns[3])),
    ],
)
def test_1d_t2_forward_CPU(targets: np.ndarray):
    """
    Test type 2 API against existing implementations by setting
    """
    N = len(targets)
    inv_targets = scipy.fft.fft(targets)
    inv_targets_tens = torch.from_numpy(inv_targets)

    assert len(inv_targets) == N

    against_torch = torch.fft.ifft(inv_targets_tens)
    against_scipy = scipy.fft.ifft(inv_targets)
    against_numpy = np.fft.ifft(inv_targets)

    assert torch.norm(against_torch - against_scipy) / N == pytest.approx(0)
    assert torch.norm(against_torch - against_numpy) / N == pytest.approx(0)

    finufft_out = (
        pytorch_finufft.functional.finufft1D2.forward(
            2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64),
            inv_targets_tens,
            isign=1,
            modeord=1,
        )
        / N
    )

    print(inv_targets)
    print(against_torch)
    print(finufft_out)

    assert torch.norm(finufft_out - np.array(targets)) / N == pytest.approx(0)
    assert torch.norm(finufft_out - against_torch) / N == pytest.approx(0)


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
