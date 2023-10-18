import numpy as np
import pytest
import torch

import pytorch_finufft


def check_t2_forward(N: int, dim: int, device: str, fftshift=False) -> None:
    slices = tuple(slice(None, N) for _ in range(dim))
    g = np.mgrid[slices] * 2 * np.pi / N
    points = torch.from_numpy(g.reshape(g.shape[0], -1)).to(device)

    targets = torch.randn(*g[0].shape, dtype=torch.complex128).to(device)

    print("N is " + str(N))
    print("dim is " + str(dim))
    print("shape of points is " + str(points.shape))
    print("shape of targets is " + str(targets.shape))

    finufft_out = pytorch_finufft.functional.finufft_type2(
        points,
        targets,
        modeord=int(not fftshift),
    )

    if fftshift:
        targets = torch.fft.ifftshift(targets)

    against_torch = torch.fft.fftn(targets)

    abs_errors = torch.abs(finufft_out - against_torch.ravel())
    l_inf_error = abs_errors.max()
    l_2_error = torch.sqrt(torch.sum(abs_errors**2))
    l_1_error = torch.sum(abs_errors)

    assert l_inf_error < 4.5e-5 * N**1.1
    assert l_2_error < 6e-5 * N**2.1
    assert l_1_error < 1.2e-4 * N**3.2


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
def test_t2_1D_forward_CPU(N: int) -> None:
    check_t2_forward(N, 1, "cpu")


@pytest.mark.parametrize("N", Ns)
def test_t2_1D_forward_cuda(N: int) -> None:
    check_t2_forward(N, 1, "cuda")


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
@pytest.mark.parametrize("fftshift", [False, True])
def test_t2_2D_forward_CPU(N: int, fftshift: bool) -> None:
    check_t2_forward(N, 2, "cpu", fftshift)


@pytest.mark.parametrize("N", Ns)
def test_t2_2D_forward_cuda(N: int) -> None:
    check_t2_forward(N, 2, "cuda")


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
def test_t2_3D_forward_CPU(N: int) -> None:
    check_t2_forward(N, 3, "cpu")


@pytest.mark.parametrize("N", Ns)
def test_t2_3D_forward_cuda(N: int) -> None:
    check_t2_forward(N, 3, "cuda")
