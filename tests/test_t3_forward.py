import numpy as np
import pytest
import torch

import pytorch_finufft

torch.manual_seed(45678)


def check_t3_forward(N: int, dim: int, device: str) -> None:
    """
    Tests against implementations of the FFT by setting up a uniform grid
    over which to call FINUFFT through the API.
    """

    slices = tuple(slice(None, N) for _ in range(dim))
    g = np.mgrid[slices] * 2 * np.pi / N
    points = torch.from_numpy(g.reshape(dim, -1)).to(device)
    values = torch.randn(*g[0].shape, dtype=torch.complex128).to(device)
    targets = (
        torch.from_numpy(np.mgrid[slices].astype(np.float64))
        .reshape(dim, -1)
        .to(device)
    )

    print("N is " + str(N))
    print("dim is " + str(dim))
    print("shape of points is " + str(points.shape))
    print("shape of values is " + str(values.shape))
    print("shape of targets is " + str(targets.shape))

    finufft_out = pytorch_finufft.functional.finufft_type3(
        points, values.flatten(), targets, eps=1e-7
    )

    against_torch = torch.fft.fftn(values)

    abs_errors = torch.abs(finufft_out - against_torch.flatten())
    l_inf_error = abs_errors.max()
    l_2_error = torch.sqrt(torch.sum(abs_errors**2))
    l_1_error = torch.sum(abs_errors)

    assert l_inf_error < 5e-5 * N**1.5
    assert l_2_error < 1.5e-5 * N**3.2
    assert l_1_error < 1.5e-5 * N**4.5


Ns_and_dims = [
    (2, 1),
    (3, 1),
    (5, 1),
    (10, 1),
    (100, 1),
    (101, 1),
    (1000, 1),
    (10001, 1),
    (2, 2),
    (3, 2),
    (5, 2),
    (10, 2),
    (101, 2),
    (2, 3),
    (3, 3),
    (5, 3),
    (10, 3),
    (37, 3),
]


@pytest.mark.parametrize("N, dim", Ns_and_dims)
def test_t3_forward_CPU(N, dim) -> None:
    check_t3_forward(N, dim, "cpu")
