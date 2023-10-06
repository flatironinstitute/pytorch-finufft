import numpy as np
import pytest
import torch
torch.manual_seed(0)

import pytorch_finufft

# Case generation
Ns = [
    10,
    15,
    75,
    76,
    95,
    96,
    100,
    101,
    1000,
    1001,
]


# Tests
@pytest.mark.parametrize("N", Ns)
def test_2d_t1_forward_CPU(N: int) -> None:
    """
    Tests against implementations of the FFT by setting up a uniform grid
    over which to call FINUFFT through the API.
    """
    g = np.mgrid[:N, :N] * 2 * np.pi / N
    x, y = g.reshape(2, -1)

    values = torch.randn(*x.shape, dtype=torch.complex128)

    print("N is " + str(N))
    print("shape of x is " + str(*x.shape))
    print("shape of y is " + str(*y.shape))
    print("shape of values is " + str(values.shape))

    finufft_out = pytorch_finufft.functional.finufft2D1.apply(
        torch.from_numpy(x),
        torch.from_numpy(y),
        values,
        N,
    )

    against_torch = torch.fft.fft2(values.reshape(g[0].shape))

    abs_errors = torch.abs(finufft_out - against_torch)
    l_inf_error = abs_errors.max()
    l_2_error = torch.sqrt(torch.sum(abs_errors**2))
    l_1_error = torch.sum(abs_errors)

    assert l_inf_error < 5e-5 * N
    assert l_2_error < 1e-5 * N ** 2
    assert l_1_error < 1e-5 * N ** 3


@pytest.mark.parametrize("N", Ns)
def test_2d_t2_forward_CPU(N: int) -> None:
    """
    Tests against implementations of the FFT by setting up a uniform grid
    over which to call FINUFFT through the API.
    """
    # Double precision test
    g = np.mgrid[:N, :N] * 2 * np.pi / N
    x, y = g.reshape(2, -1)

    values = torch.randn(*g[0].shape, dtype=torch.complex128)

    finufft_out = (
        pytorch_finufft.functional.finufft2D2.apply(
            torch.from_numpy(x), torch.from_numpy(y), values
        )
    ).reshape(g[0].shape) / (N**2)

    against_torch = torch.fft.ifft2(values)

    assert abs((finufft_out - against_torch).sum()) / (N**3) == pytest.approx(
        0, abs=1e-6
    )

    g = np.mgrid[:N, :N] * 2 * np.pi / N
    x, y = g.reshape(2, -1)

    # single precision test
    values = torch.randn(*g[0].shape, dtype=torch.complex64)

    finufft_out = (
        pytorch_finufft.functional.finufft2D2.apply(
            torch.from_numpy(x).to(torch.float32),
            torch.from_numpy(y).to(torch.float32),
            values,
        )
    ).reshape(g[0].shape) / (N**2)

    against_torch = torch.fft.ifft2(values)

    abs_errors = torch.abs(finufft_out - against_torch)
    l_inf_error = abs_errors.max()
    l_2_error = torch.sqrt(torch.sum(abs_errors**2))
    l_1_error = torch.sum(abs_errors)

    assert l_inf_error < 1e-5 * N
    assert l_2_error < 1e-5 * N ** 2
    assert l_1_error < 1e-5 * N ** 3


# @pytest.mark.parametrize("N", Ns)
# def test_2d_t3_forward_CPU(N: int) -> None:
#     g = np.mgrid[:3, :3] * 2 * np.pi / 3
#     x, y = g.reshape(2, -1)

#     values = torch.randn(*g[0].shape, dtype=torch.complex128).numpy()

#     f = finufft.nufft2d2(x, y, np.fft.fftshift(values)).reshape(g[0].shape) / 9

#     comparison = np.fft.ifft2(values)

#     assert abs((f - comparison).sum()) / (N**3) == pytest.approx(0, abs=1e-6)

#     pass


@pytest.mark.parametrize("N", Ns)
def test_t1_forward_CPU(N: int) -> None:
    """
    Tests against implementations of the FFT by setting up a uniform grid
    over which to call FINUFFT through the API.
    """
    g = np.mgrid[:N, :N] * 2 * np.pi / N
    points = torch.from_numpy(g.reshape(2, -1))

    values = torch.randn(*points[0].shape, dtype=torch.complex128)

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
    assert l_2_error < 1e-5 * N ** 2
    assert l_1_error < 1e-5 * N ** 3

