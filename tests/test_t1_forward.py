import numpy as np
import pytest
import torch

import pytorch_finufft


def check_t1_forward(N: int, dim: int, device: str) -> None:
    """
    Tests against implementations of the FFT by setting up a uniform grid
    over which to call FINUFFT through the API.
    """
    slices = tuple(slice(None, N) for _ in range(dim))
    g = np.mgrid[slices] * 2 * np.pi / N
    points = torch.from_numpy(g.reshape(dim, -1)).to(device)

    values = torch.randn(*points[0].shape, dtype=torch.complex128).to(device)

    print("N is " + str(N))
    print("dim is " + str(dim))
    print("shape of points is " + str(points.shape))
    print("shape of values is " + str(values.shape))

    finufft_out = pytorch_finufft.functional.finufft_type1(
        points,
        values,
        tuple(N for _ in range(dim)),
    )

    against_torch = torch.fft.fftn(values.reshape(g[0].shape))

    abs_errors = torch.abs(finufft_out - against_torch)
    l_inf_error = abs_errors.max()
    l_2_error = torch.sqrt(torch.sum(abs_errors**2))
    l_1_error = torch.sum(abs_errors)

    assert l_inf_error < 1.5e-5 * N**1.5
    assert l_2_error < 1e-5 * N**3
    assert l_1_error < 1e-5 * N**4.5


#### 1D TESTS ####

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


@pytest.mark.parametrize("N", Ns)
def test_t1_1D_forward_CPU(N: int) -> None:
    check_t1_forward(N, 1, "cpu")


@pytest.mark.parametrize("N", Ns)
def test_t1_1D_forward_cuda(N: int) -> None:
    check_t1_forward(N, 1, "cuda")


#### 2D TESTS ####
Ns = [
    3,
    10,
    15,
    75,
    76,
    95,
    96,
    100,
    101,
]


@pytest.mark.parametrize("N", Ns)
def test_t1_2D_forward_CPU(N: int) -> None:
    check_t1_forward(N, 2, "cpu")


@pytest.mark.parametrize("N", Ns)
def test_t1_2D_forward_cuda(N: int) -> None:
    check_t1_forward(N, 2, "cuda")


#### 3D TESTS ####

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


@pytest.mark.parametrize("N", Ns)
def test_t1_3D_forward_CPU(N: int) -> None:
    check_t1_forward(N, 3, "cpu")


@pytest.mark.parametrize("N", Ns)
def test_t1_3D_forward_cuda(N: int) -> None:
    check_t1_forward(N, 3, "cuda")
