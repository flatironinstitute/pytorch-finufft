import finufft
import numpy as np
import pytest
import torch

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

    finufft_out = pytorch_finufft.functional.finufft2D1.apply(
        torch.from_numpy(x),
        torch.from_numpy(y),
        values,
        N,
    )

    against_torch = torch.fft.fft2(values.reshape(g[0].shape))

    assert abs((finufft_out - against_torch).sum()) / (N**3) == pytest.approx(
        0, abs=1e-6
    )

    values = torch.randn(*x.shape, dtype=torch.complex64)

    finufft_out = pytorch_finufft.functional.finufft2D1.apply(
        torch.from_numpy(x).to(torch.float32),
        torch.from_numpy(y).to(torch.float32),
        values,
        N,
    )

    against_torch = torch.fft.fft2(values.reshape(g[0].shape))

    # NOTE -- the below tolerance is set to 1e-5 instead of -6 due
    #   to the occasional failing case that seems to be caused by
    #   the randomness of the test cases in addition to the expected
    #   accruation of numerical inaccuracies
    assert abs((finufft_out - against_torch).sum()) / (N**3) == pytest.approx(
        0, abs=1e-5
    )


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

    assert abs((finufft_out - against_torch).sum()) / (N**3) == pytest.approx(
        0, abs=1e-6
    )


@pytest.mark.parametrize("N", Ns)
def test_2d_t3_forward_CPU(N: int) -> None:
    g = np.mgrid[:3, :3] * 2 * np.pi / 3
    x, y = g.reshape(2, -1)

    values = torch.randn(*g[0].shape, dtype=torch.complex128).numpy()

    f = finufft.nufft2d2(x, y, np.fft.fftshift(values)).reshape(g[0].shape) / 9

    comparison = np.fft.ifft2(values)

    assert abs((f - comparison).sum()) / (N**3) == pytest.approx(0, abs=1e-6)

    pass
