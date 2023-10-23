from typing import Tuple

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

import pytorch_finufft

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)


def check_t2_backward(
    targets_shape: Tuple[int, ...],
    n_points: int,
    fftshift: bool,
    isign: int,
    device: str,
    points_or_targets: bool,
) -> None:
    dims = len(targets_shape)
    points_shape = (dims, n_points)
    points = torch.rand(points_shape, dtype=torch.float64).to(device) * 2 * np.pi
    targets = torch.randn(*targets_shape, dtype=torch.complex128).to(device)

    points.requires_grad = points_or_targets
    targets.requires_grad = not points_or_targets

    inputs = (points, targets)

    def func(points, targets):
        return pytorch_finufft.functional.finufft_type2(
            points,
            targets,
            modeord=int(not fftshift),
            isign=isign,
        )

    assert gradcheck(
        func, inputs, eps=1e-8, atol=3e-4 * np.prod(targets_shape) ** (1.0 / dims)
    )


shapes_and_Ns = [
    ((2,), 2),
    ((2,), 51),
    ((5,), 4),
    ((6,), 50),
    ((101,), 10),
    ((2, 2), 21),
    ((20, 21), 51),
    ((8, 30), 23),
    ((5, 4, 3), 10),
]


@pytest.mark.parametrize("target_shape, N", shapes_and_Ns)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_backward_CPU_points(target_shape, N, fftshift, isign) -> None:
    check_t2_backward(target_shape, N, fftshift, isign, "cpu", True)


@pytest.mark.parametrize("target_shape, N", shapes_and_Ns)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_backward_CPU_values(target_shape, N, fftshift, isign) -> None:
    check_t2_backward(target_shape, N, fftshift, isign, "cpu", False)


@pytest.mark.parametrize("target_shape, N", shapes_and_Ns)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_backward_cuda_points(target_shape, N, fftshift, isign) -> None:
    check_t2_backward(target_shape, N, fftshift, isign, "cuda", True)


@pytest.mark.parametrize("target_shape, N", shapes_and_Ns)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_backward_cuda_values(target_shape, N, fftshift, isign) -> None:
    check_t2_backward(target_shape, N, fftshift, isign, "cuda", False)
