from typing import Any, Callable, Tuple, Union

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

import pytorch_finufft

torch.random.manual_seed(1234)


def check_t2_batched_targets(
    F: Callable[..., Any],
    N: int,
    batchsize: Union[int, Tuple[int, ...]],
    dim: int,
    device: str,
) -> None:
    if not isinstance(batchsize, tuple):
        batchsize = (batchsize,)

    slices = tuple(slice(None, N) for _ in range(dim))
    g = np.mgrid[slices] * 2 * np.pi / N
    points = torch.from_numpy(g.reshape(g.shape[0], -1)).to(device)

    targets = torch.randn(*batchsize, *g[0].shape, dtype=torch.complex128).to(device)

    print("N is " + str(N))
    print("dim is " + str(dim))
    print("shape of points is " + str(points.shape))
    print("shape of targets is " + str(targets.shape))

    finufft_out = F(
        points,
        targets,
    )

    against_torch = torch.fft.fftn(targets, dim=tuple(range(-dim, 0)))

    abs_errors = torch.abs(finufft_out - against_torch.reshape(finufft_out.shape))
    l_inf_error = abs_errors.max()
    l_2_error = torch.sqrt(torch.sum(abs_errors**2))
    l_1_error = torch.sum(abs_errors)

    assert l_inf_error < 4.5e-5 * N**1.1
    assert l_2_error < 6e-5 * N**2.1
    assert l_1_error < 1.2e-4 * N**3.2

    points.requires_grad = True
    targets.requires_grad = True

    gradcheck(F, (points, targets), eps=1e-8, atol=2e-4)


targets_cases = [
    (2, 1, 1),  # check that batch of 1 is happy
    (2, (2, 3), 1),
    (2, 2, 2),
    (2, (2, 1, 3), 3),
]


@pytest.mark.parametrize("N, batch, dim", targets_cases)
def test_t2_batching_CPU(N, batch, dim):
    check_t2_batched_targets(
        pytorch_finufft.functional.finufft_type2, N, batch, dim, "cpu"
    )


@pytest.mark.parametrize("N, batch, dim", targets_cases)
def test_t2_batching_cuda(N, batch, dim):
    check_t2_batched_targets(
        pytorch_finufft.functional.finufft_type2, N, batch, dim, "cuda"
    )


def batch_vmapped(batch: Union[int, Tuple[int, ...]]) -> Callable[..., Any]:
    if not isinstance(batch, tuple):
        batch = (batch,)

    F = pytorch_finufft.functional.finufft_type2
    for _ in batch:
        F = torch.vmap(F, in_dims=(None, 0), out_dims=0)
    return F


@pytest.mark.parametrize("N, batch, dim", targets_cases)
def test_t2_vmap_targets_CPU(N, batch, dim):
    check_t2_batched_targets(batch_vmapped(batch), N, batch, dim, "cpu")


@pytest.mark.parametrize("N, batch, dim", targets_cases)
def test_t2_vmap_targets_cuda(N, batch, dim):
    check_t2_batched_targets(batch_vmapped(batch), N, batch, dim, "cuda")


# because points are not natively batchable in finufft, we only test vmap
def check_t2_vmapped_points(
    N: int,
    targets_batchsize: Union[int, Tuple],
    dim: int,
    device: str,
):
    if not isinstance(targets_batchsize, tuple):
        targets_batchsize = (targets_batchsize,)

    slices = tuple(slice(None, N) for _ in range(dim))
    g = np.mgrid[slices] * 2 * np.pi / N
    points = torch.from_numpy(g.reshape(g.shape[0], -1)).to(device)

    targets = torch.randn(*targets_batchsize, *g[0].shape, dtype=torch.complex128).to(
        device
    )
    points = torch.stack(
        (points, points + 0.02), dim=0
    )  # slight perturbation to check that vmap is working

    print("N is " + str(N))
    print("dim is " + str(dim))
    print("shape of points is " + str(points.shape))
    print("shape of targets is " + str(targets.shape))

    F = torch.vmap(
        pytorch_finufft.functional.finufft_type2,
        in_dims=(0, 0 if targets_batchsize else None),
        out_dims=0,
    )

    finufft_out = F(
        points,
        targets,
    )

    against_torch = torch.fft.fftn(targets, dim=tuple(range(-dim, 0)))

    if targets_batchsize:
        against_torch = against_torch[0]
    abs_errors = torch.abs(finufft_out[0].ravel() - against_torch.ravel())
    l_inf_error = abs_errors.max()
    l_2_error = torch.sqrt(torch.sum(abs_errors**2))
    l_1_error = torch.sum(abs_errors)

    assert l_inf_error < 4.5e-5 * N**1.1
    assert l_2_error < 6e-5 * N**2.1
    assert l_1_error < 1.2e-4 * N**3.2

    points.requires_grad = True
    targets.requires_grad = True

    gradcheck(F, (points, targets), eps=1e-8, atol=2e-4)


points_cases = [
    (2, (), 1),
    (2, (), 2),
    (2, (), 3),
    (2, (2,), 1),
    (2, (2,), 2),
    (2, (2,), 3),
]


@pytest.mark.parametrize("N, batch, dim", points_cases)
def test_t2_vmap_points_CPU(N, batch, dim):
    check_t2_vmapped_points(N, batch, dim, "cpu")


@pytest.mark.parametrize("N, batch, dim", points_cases)
def test_t2_vmap_points_cuda(N, batch, dim):
    check_t2_vmapped_points(N, batch, dim, "cuda")
