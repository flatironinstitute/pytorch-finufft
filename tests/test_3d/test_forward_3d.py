import numpy as np
import pytest
import torch

import pytorch_finufft

# Case generation
Ns = [
    5,
    10,
    15,
    16,
    25,
    26,
    37,
    100,
    101,
]


# Tests
@pytest.mark.parametrize("N", Ns)
def test_3d_t1_forward_CPU(N: int) -> None:
    """
    Tests against implementations of the FFT by setting up a uniform grid
    over which to call FINUFFT through the API.
    """
    for _ in range(5):
        g = np.mgrid[:N, :N, :N] * 2 * np.pi / N
        x, y, z = g.reshape(3, -1)

        values = torch.randn(*x.shape, dtype=torch.complex128)

        finufft_out = pytorch_finufft.functional.finufft3D1.apply(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(z),
            values,
            N,
        )

        against_torch = torch.fft.fftn(values.reshape(g[0].shape))

        abs_errors = torch.abs(finufft_out - against_torch)
        l_inf_error = abs_errors.max()
        l_2_error = torch.sqrt(torch.sum(abs_errors**2))
        l_1_error = torch.sum(abs_errors)

        assert l_inf_error < 2e-5 * N ** 1.5
        assert l_2_error < 1e-5 * N ** 3
        assert l_1_error < 1e-5 * N ** 4.5



@pytest.mark.parametrize("N", Ns)
def test_3d_t2_forward_CPU(N: int) -> None:
    """
    Tests against implementations of the FFT by setting up a uniform grid
    over which to call FINUFFT through the API
    """
    # Double precision test

    for _ in range(5):
        g = np.mgrid[:N, :N, :N] * 2 * np.pi / N
        x, y, z = g.reshape(3, -1)

        values = torch.randn(*g[0].shape, dtype=torch.complex128)

        finufft_out = pytorch_finufft.functional.finufft3D2.apply(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(z),
            values,
        ).reshape(g[0].shape) / (N**3)

        against_torch = torch.fft.ifftn(values)

        abs_errors = torch.abs(finufft_out - against_torch)
        l_inf_error = abs_errors.max()
        l_2_error = torch.sqrt(torch.sum(abs_errors**2))
        l_1_error = torch.sum(abs_errors)

        assert l_inf_error < 1e-5 * N ** 1.5
        assert l_2_error < 1e-5 * N ** 3
        assert l_1_error < 1e-5 * N ** 4.5
