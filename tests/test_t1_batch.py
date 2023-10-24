from typing import Any, Callable, Tuple, Union

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

import pytorch_finufft

torch.random.manual_seed(1234)


def check_t1_batched_targets(
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
    points = torch.from_numpy(g.reshape(dim, -1)).to(device)

    values = torch.randn(*batchsize, *points[0].shape, dtype=torch.complex128).to(
        device
    )

    print("N is " + str(N))
    print("dim is " + str(dim))
    print("shape of points is " + str(points.shape))
    print("shape of values is " + str(values.shape))

    output_shape = tuple(N for _ in range(dim))

    finufft_out = F(
        points,
        values,
        output_shape,
    )

    against_torch = torch.fft.fftn(
        values.reshape(*batchsize, *g[0].shape), dim=tuple(range(-dim, 0))
    )

    abs_errors = torch.abs(finufft_out - against_torch.reshape(finufft_out.shape))
    l_inf_error = abs_errors.max()
    l_2_error = torch.sqrt(torch.sum(abs_errors**2))
    l_1_error = torch.sum(abs_errors)

    assert l_inf_error < 4.5e-5 * N**1.1
    assert l_2_error < 6e-5 * N**2.1
    assert l_1_error < 1.2e-4 * N**3.2

    points.requires_grad = True
    values.requires_grad = True

    def f(p, v):
        return F(p, v, output_shape)

    gradcheck(f, (points, values), eps=1e-8, atol=2e-4)


values_cases = [
    (2, 1, 1),  # check that batch of 1 is happy
    (2, (2, 3), 1),
    (2, 2, 2),
    (2, (2, 1, 3), 3),
]


@pytest.mark.parametrize("N, batch, dim", values_cases)
def test_t1_batching_CPU(N, batch, dim):
    check_t1_batched_targets(
        pytorch_finufft.functional.finufft_type1, N, batch, dim, "cpu"
    )


@pytest.mark.parametrize("N, batch, dim", values_cases)
def test_t1_batching_cuda(N, batch, dim):
    check_t1_batched_targets(
        pytorch_finufft.functional.finufft_type1, N, batch, dim, "cuda"
    )


def batch_vmapped(batch: Union[int, Tuple[int, ...]]) -> Callable[..., Any]:
    if not isinstance(batch, tuple):
        batch = (batch,)

    F = pytorch_finufft.functional.finufft_type1
    for _ in batch:
        F = torch.vmap(F, in_dims=(None, 0, None), out_dims=0)
    return F


@pytest.mark.parametrize("N, batch, dim", values_cases)
def test_t1_vmap_targets_CPU(N, batch, dim):
    check_t1_batched_targets(batch_vmapped(batch), N, batch, dim, "cpu")


@pytest.mark.parametrize("N, batch, dim", values_cases)
def test_t1_vmap_targets_cuda(N, batch, dim):
    check_t1_batched_targets(batch_vmapped(batch), N, batch, dim, "cuda")


# because points are not natively batchable in finufft, we only test vmap
def check_t1_vmapped_points(
    N: int,
    values_batchsize: Union[int, Tuple],
    dim: int,
    device: str,
):
    if not isinstance(values_batchsize, tuple):
        values_batchsize = (values_batchsize,)

    slices = tuple(slice(None, N) for _ in range(dim))
    g = np.mgrid[slices] * 2 * np.pi / N
    points = torch.from_numpy(g.reshape(dim, -1)).to(device)

    values = torch.randn(
        *values_batchsize, *points[0].shape, dtype=torch.complex128
    ).to(device)
    points = torch.stack(
        (points, points + 0.02), dim=0
    )  # slight perturbation to check that vmap is working

    print("N is " + str(N))
    print("dim is " + str(dim))
    print("shape of points is " + str(points.shape))
    print("shape of values is " + str(values.shape))

    output_shape = tuple(N for _ in range(dim))

    F = torch.vmap(
        pytorch_finufft.functional.finufft_type1,
        in_dims=(0, 0 if values_batchsize else None, None),
        out_dims=0,
    )

    finufft_out = F(
        points,
        values,
        output_shape,
    )

    against_torch = torch.fft.fftn(
        values.reshape(*values_batchsize, *g[0].shape), dim=tuple(range(-dim, 0))
    )
    if values_batchsize:
        against_torch = against_torch[0]
    abs_errors = torch.abs(finufft_out[0].ravel() - against_torch.ravel())
    l_inf_error = abs_errors.max()
    l_2_error = torch.sqrt(torch.sum(abs_errors**2))
    l_1_error = torch.sum(abs_errors)

    assert l_inf_error < 4.5e-5 * N**1.1
    assert l_2_error < 6e-5 * N**2.1
    assert l_1_error < 1.2e-4 * N**3.2

    points.requires_grad = True
    values.requires_grad = True

    def f(p, v):
        return F(p, v, output_shape)

    gradcheck(f, (points, values), eps=1e-8, atol=2e-4)


points_cases = [
    (2, (), 1),
    (2, (), 2),
    (2, (), 3),
    (2, (2,), 1),
    (2, (2,), 2),
    (2, (2,), 3),
]


@pytest.mark.parametrize("N, batch, dim", points_cases)
def test_t1_vmap_points_CPU(N, batch, dim):
    check_t1_vmapped_points(N, batch, dim, "cpu")


@pytest.mark.parametrize("N, batch, dim", points_cases)
def test_t1_vmap_points_cuda(N, batch, dim):
    check_t1_vmapped_points(N, batch, dim, "cuda")
