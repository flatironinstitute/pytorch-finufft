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
def test_t2_forward_CPU(N, dim) -> None:
    check_t2_forward(N, dim, "cpu")


@pytest.mark.parametrize("N, dim", Ns_and_dims)
def test_t2_forward_cuda(N, dim) -> None:
    check_t2_forward(N, dim, "cuda")
