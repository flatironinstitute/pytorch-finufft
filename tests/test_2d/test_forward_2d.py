import numpy as np
import pytest
import scipy
import torch
from numpy.random import standard_normal
import finufft

import pytorch_finufft


# Case generation
Ns = [
    10,
    15,
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

    f = finufft.nufft2d1(x, y, values.numpy(), N, modeord=1)

    finufft_out = pytorch_finufft.functional.finufft2D1.apply(
        torch.from_numpy(x),
        torch.from_numpy(y),
        values,
        N,
    )

    against_np = np.fft.fft2(values.numpy().reshape(g[0].shape))
    against_torch = torch.fft.fft2(values.reshape(g[0].shape))

    # assert abs((f - against_np).sum()) / (N**3) == pytest.approx(0, abs=1e-6)
    # assert abs((f - against_torch.numpy()).sum()) / (N**3) == pytest.approx(0, abs=1e-6)

    assert abs((finufft_out - against_torch).sum()) / (N**3) == pytest.approx(0, abs=1e-6)


def asdftest_2d_t2_forward_CPU(targets: torch.Tensor) -> None:

    pass


def asdftest_2d_t3_forward_CPU(values: torch.Tensor) -> None:

    pass
