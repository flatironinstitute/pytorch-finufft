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
def test_t1_forward_CPU(N, dim) -> None:
    check_t1_forward(N, dim, "cpu")


@pytest.mark.parametrize("N, dim", Ns_and_dims)
def test_t1_forward_cuda(N, dim) -> None:
    check_t1_forward(N, dim, "cuda")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="require multiple GPUs")
def test_t1_forward_cuda_device_1() -> None:
    # added after https://github.com/flatironinstitute/pytorch-finufft/issues/103
    check_t1_forward(3, 1, "cuda:1")
