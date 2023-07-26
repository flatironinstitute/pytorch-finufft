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
    g = np.mgrid[:N, :N, :N] * 2 * np.pi / N
    x, y, z = g.reshape(3, -1)

    values = torch.randn(*x.shape, dtype=torch.complex128).numpy()

    f = finufft.nufft3d1(x, y, z, values, N, modeord=1)

    comparison = np.fft.fftn(values.reshape(g[0].shape))

    assert abs((f - comparison).sum()) / (N**4) == pytest.approx(0, abs=1e-6)
