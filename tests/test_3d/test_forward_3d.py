import finufft
import numpy as np
import pytest
import scipy
import torch
from numpy.random import standard_normal

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

        assert abs((finufft_out - against_torch).sum()) / (N**4) == pytest.approx(
            0, abs=1e-6
        )


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

        assert (abs((finufft_out - against_torch).sum())) / (N**4) == pytest.approx(
            0, abs=1e-6
        )
