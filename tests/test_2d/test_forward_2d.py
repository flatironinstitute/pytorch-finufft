import numpy as np
import pytest
import torch

import pytorch_finufft

torch.manual_seed(1234)


# Case generation
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


def check_t1_forward(N: int, device: str) -> None:
    """
    Tests against implementations of the FFT by setting up a uniform grid
    over which to call FINUFFT through the API.
    """
    g = np.mgrid[:N, :N] * 2 * np.pi / N
    points = torch.from_numpy(g.reshape(2, -1)).to(device)

    values = torch.randn(*points[0].shape, dtype=torch.complex128).to(device)

    print("N is " + str(N))
    print("shape of points is " + str(points.shape))
    print("shape of values is " + str(values.shape))

    finufft_out = pytorch_finufft.functional.finufft_type1.apply(
        points,
        values,
        (N, N),
    )

    against_torch = torch.fft.fft2(values.reshape(g[0].shape))

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


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("fftshift", [False, True])
def test_t2_forward_CPU(N: int, fftshift: bool) -> None:
    """
    Tests against implementations of the FFT by setting up a uniform grid
    over which to call FINUFFT through the API.
    """
    g = np.mgrid[:N, :N] * 2 * np.pi / N
    points = torch.from_numpy(g.reshape(2, -1))

    targets = torch.randn(*g[0].shape, dtype=torch.complex128)

    print("N is " + str(N))
    print("shape of points is " + str(points.shape))
    print("shape of targets is " + str(targets.shape))

    finufft_out = pytorch_finufft.functional.finufft_type2.apply(
        points,
        targets,
        {"modeord": int(not fftshift)},
    )

    if fftshift:
        against_torch = torch.fft.fft2(torch.fft.ifftshift(targets))
    else:
        against_torch = torch.fft.fft2(targets)

    abs_errors = torch.abs(finufft_out - against_torch.ravel())
    l_inf_error = abs_errors.max()
    l_2_error = torch.sqrt(torch.sum(abs_errors**2))
    l_1_error = torch.sum(abs_errors)

    assert l_inf_error < 4.5e-5 * N
    assert l_2_error < 1e-5 * N**2
    assert l_1_error < 1e-5 * N**3
