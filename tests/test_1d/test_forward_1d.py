import numpy as np
import pytest
import torch

import pytorch_finufft

# Case generation
Ns = [
    5,
    10,
    15,
    100,
    101,
    1000,
    1001,
    3750,
    5000,
    5001,
    6250,
    7500,
]


def check_t1_forward(N: int, device: str) -> None:
    """
    Tests against implementations of the FFT by setting up a uniform grid
    over which to call FINUFFT through the API.
    """
    g = np.mgrid[:N] * 2 * np.pi / N
    g.shape = 1, -1
    points = torch.from_numpy(g.reshape(1, -1)).to(device)

    values = torch.randn(*points[0].shape, dtype=torch.complex128).to(device)

    print("N is " + str(N))
    print("shape of points is " + str(points.shape))
    print("shape of values is " + str(values.shape))

    finufft_out = pytorch_finufft.functional.finufft_type1.apply(
        points,
        values,
        (N,),
    )

    against_torch = torch.fft.fft(values.reshape(g[0].shape))

    abs_errors = torch.abs(finufft_out - against_torch)
    l_inf_error = abs_errors.max()
    l_2_error = torch.sqrt(torch.sum(abs_errors**2))
    l_1_error = torch.sum(abs_errors)

    assert l_inf_error < 4.5e-5 * N
    assert l_2_error < 1e-5 * N**2
    assert l_1_error < 1e-5 * N**3


@pytest.mark.parametrize("N", Ns)
def test_t1_forward_CPU(N: int) -> None:
    check_t1_forward(N, "cpu")


@pytest.mark.parametrize("N", Ns)
def test_t1_forward_cuda(N: int) -> None:
    check_t1_forward(N, "cuda")


cases = [torch.tensor([1.0, 2.5, -1.0, -1.5, 1.5], dtype=torch.complex128)]
for n in Ns:
    cases.append(
        torch.randn(n, dtype=torch.float64) + 1j * torch.randn(n, dtype=torch.float64)
    )
    cases.append(
        torch.randn(n, dtype=torch.float32) + 1j * torch.randn(n, dtype=torch.float32)
    )


@pytest.mark.parametrize("targets", cases)
def test_1d_t2_forward_CPU(targets: torch.Tensor):
    """
    Test type 2 API against existing implementations by setting
    """
    N = len(targets)
    inv_targets = torch.fft.fft(targets)
    assert len(inv_targets) == N

    against_torch = torch.fft.ifft(inv_targets)

    data_type = torch.float64 if targets.dtype is torch.complex128 else torch.float32

    finufft_out = (
        pytorch_finufft.functional.finufft1D2.apply(
            2 * np.pi * torch.arange(0, 1, 1 / N, dtype=data_type),
            inv_targets,
        )
        / N
    )

    assert torch.norm(finufft_out - np.array(targets)) / N**2 == pytest.approx(
        0, abs=1e-05
    )
    assert torch.norm(finufft_out - against_torch) / N**2 == pytest.approx(
        0, abs=1e-05
    )


@pytest.mark.parametrize("N", Ns)
def test_t2_forward_CPU(N: int) -> None:
    """
    Tests against implementations of the FFT by setting up a uniform grid
    over which to call FINUFFT through the API.
    """
    g = np.mgrid[:N,] * 2 * np.pi / N
    points = torch.from_numpy(g.reshape(g.shape[0], -1))

    targets = torch.randn(*g[0].shape, dtype=torch.complex128)

    print("N is " + str(N))
    print("shape of points is " + str(points.shape))
    print("shape of targets is " + str(targets.shape))

    finufft_out = pytorch_finufft.functional.finufft_type2.apply(
        points,
        targets,
    )

    against_torch = torch.fft.fftn(targets)

    abs_errors = torch.abs(finufft_out - against_torch.ravel())
    l_inf_error = abs_errors.max()
    l_2_error = torch.sqrt(torch.sum(abs_errors**2))
    l_1_error = torch.sum(abs_errors)

    assert l_inf_error < 4.5e-5 * N ** 1.1
    assert l_2_error < 6e-5 * N ** 2.1
    assert l_1_error < 1.2e-4 * N ** 3.2

