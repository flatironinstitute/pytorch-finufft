import numpy as np
import pytest
import scipy
import torch
from numpy.testing import assert_almost_equal

import pytorch_finufft

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
cases = [torch.tensor([1.0, 2.5, -1.0, -1.5, 1.5], dtype=torch.complex128)]
for n in Ns:
    cases.append(
        torch.randn(n, dtype=torch.float64)
        + 1j * torch.randn(n, dtype=torch.float64)
    )
    cases.append(
        torch.randn(n, dtype=torch.float32)
        + 1j * torch.randn(n, dtype=torch.float32)
    )


# Tests
@pytest.mark.parametrize("values", cases)
def test_1d_t1_forward_CPU(values: torch.Tensor) -> None:
    """
    Tests against implementations of the FFT by setting up a uniform grid over
    which to call FINUFFT through the API.
    """
    N = len(values)
    val_np = values.numpy()

    against_torch = torch.fft.fft(values)
    against_scipy = torch.tensor(scipy.fft.fft(val_np))

    data_type = (
        torch.float64 if values.dtype is torch.complex128 else torch.float32
    )

    finufft1D1_out = pytorch_finufft.functional.finufft1D1.apply(
        2 * np.pi * torch.arange(0, 1, 1 / N, dtype=data_type),
        values,
        N,
    )

    if values.dtype is torch.complex64:
        assert finufft1D1_out.dtype is torch.complex64
    else:
        assert finufft1D1_out.dtype is torch.complex128

    assert against_torch.dtype == values.dtype
    assert (
        torch.linalg.norm(finufft1D1_out - against_torch) / N**2
    ) == pytest.approx(0, abs=1e-06)
    assert (
        torch.linalg.norm(finufft1D1_out - against_scipy) / N**2
    ) == pytest.approx(0, abs=1e-06)


@pytest.mark.parametrize("targets", cases)
def test_1d_t2_forward_CPU(targets: torch.Tensor):
    """
    Test type 2 API against existing implementations by setting
    """
    N = len(targets)
    inv_targets = torch.fft.fft(targets)
    assert len(inv_targets) == N

    against_torch = torch.fft.ifft(inv_targets)

    data_type = (
        torch.float64 if targets.dtype is torch.complex128 else torch.float32
    )

    finufft_out = (
        pytorch_finufft.functional.finufft1D2.apply(
            2 * np.pi * torch.arange(0, 1, 1 / N, dtype=data_type),
            inv_targets,
        )
        / N
    )

    assert torch.norm(
        finufft_out - np.array(targets)
    ) / N**2 == pytest.approx(0, abs=1e-05)
    assert torch.norm(finufft_out - against_torch) / N**2 == pytest.approx(
        0, abs=1e-05
    )


@pytest.mark.parametrize("values", cases)
def test_1d_t3_forward_CPU(values: torch.Tensor):
    """
    Test type 3 API against existing implementations
    """
    N = len(values)

    data_type = (
        torch.float64 if values.dtype is torch.complex128 else torch.float32
    )

    points = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=data_type)
    targets = torch.arange(0, N, dtype=data_type)

    finufft_out = pytorch_finufft.functional.finufft1D3.apply(
        points, values, targets
    )

    against_torch = torch.fft.fft(values)
    against_scipy = scipy.fft.fft(values.numpy())

    assert against_torch.dtype == finufft_out.dtype
    assert (
        torch.linalg.norm(finufft_out - against_torch) / N**2
    ) == pytest.approx(0, abs=1e-05)
    assert (
        torch.linalg.norm(finufft_out - torch.from_numpy(against_scipy))
        / N**2
    ) == pytest.approx(0, abs=1e-05)
