import numpy as np
import pytest
import torch

import pytorch_finufft

torch.manual_seed(1234)


# Case generation
Ns = [
    3,
    5,
    10,
    15,
    16,
    25,
    26,
    37,
]


def check_t1_forward(N: int, device: str) -> None:
    """
    Tests against implementations of the FFT by setting up a uniform grid
    over which to call FINUFFT through the API.
    """
    g = np.mgrid[:N, :N, :N] * 2 * np.pi / N
    points = torch.from_numpy(g.reshape(3, -1)).to(device)

    values = torch.randn(*points[0].shape, dtype=torch.complex128).to(device)

    print("N is " + str(N))
    print("shape of points is " + str(points.shape))
    print("shape of values is " + str(values.shape))

    finufft_out = pytorch_finufft.functional.finufft_type1(
        points,
        values,
        (N, N, N),
    )

    against_torch = torch.fft.fftn(values.reshape(g[0].shape))

    abs_errors = torch.abs(finufft_out - against_torch)
    l_inf_error = abs_errors.max()
    l_2_error = torch.sqrt(torch.sum(abs_errors**2))
    l_1_error = torch.sum(abs_errors)

    assert l_inf_error < 1.5e-5 * N**1.5
    assert l_2_error < 1e-5 * N**3
    assert l_1_error < 1e-5 * N**4.5


@pytest.mark.parametrize("N", Ns)
def test_t1_forward_CPU(N: int) -> None:
    check_t1_forward(N, "cpu")


@pytest.mark.parametrize("N", Ns)
def test_t1_forward_cuda(N: int) -> None:
    check_t1_forward(N, "cuda")


@pytest.mark.parametrize("N", Ns)
def test_t2_forward_CPU(N: int) -> None:
    """
    Tests against implementations of the FFT by setting up a uniform grid
    over which to call FINUFFT through the API.
    """
    g = np.mgrid[:N, :N, :N] * 2 * np.pi / N
    points = torch.from_numpy(g.reshape(g.shape[0], -1))

    targets = torch.randn(*g[0].shape, dtype=torch.complex128)

    print("N is " + str(N))
    print("shape of points is " + str(points.shape))
    print("shape of targets is " + str(targets.shape))

    finufft_out = pytorch_finufft.functional.finufft_type2(
        points,
        targets,
    )

    against_torch = torch.fft.fftn(targets)

    abs_errors = torch.abs(finufft_out - against_torch.ravel())
    l_inf_error = abs_errors.max()
    l_2_error = torch.sqrt(torch.sum(abs_errors**2))
    l_1_error = torch.sum(abs_errors)

    assert l_inf_error < 4.5e-5 * N**1.1
    assert l_2_error < 6e-5 * N**2.1
    assert l_1_error < 1.2e-4 * N**3.2
