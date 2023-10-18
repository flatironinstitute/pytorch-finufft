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
    N: int,
    points_shape: Tuple[int, ...],
    fftshift: bool,
    isign: int,
    device: str,
    points_or_targets: bool,
) -> None:
    dims = points_shape[0]
    points = torch.rand(points_shape, dtype=torch.float64).to(device) * 2 * np.pi
    targets = torch.randn(tuple(np.repeat(N, dims)), dtype=torch.complex128).to(device)

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

    assert gradcheck(func, inputs, eps=1e-8, atol=1.5e-5 * N)


#### 1D TESTS ####

Ns = [
    5,
    8,
    10,
    15,
    16,
    63,
]

length_modifiers = [-1, 0, 1, 4]


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_1D_backward_CPU_targets(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, (1, N + modifier), fftshift, isign, "cpu", False)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_1D_backward_CPU_points(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, (1, N + modifier), fftshift, isign, "cpu", True)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_1D_backward_cuda_targets(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, (1, N + modifier), fftshift, isign, "cuda", False)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_1D_backward_cuda_points(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, (1, N + modifier), fftshift, isign, "cuda", True)


#### 2D TESTS ####

Ns = [
    3,
    # 5,
    # 8,
    # 10,
    # 15,
    # 16,
]

length_modifiers = [
    0,
    1,
    # 4,
    -1,
]


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_2D_backward_CPU_targets(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, (2, N + modifier), fftshift, isign, "cpu", False)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_2D_backward_CPU_points(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, (2, N + modifier), fftshift, isign, "cpu", True)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_2D_backward_cuda_targets(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, (2, N + modifier), fftshift, isign, "cuda", False)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_2D_backward_cuda_points(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, (2, N + modifier), fftshift, isign, "cuda", True)


#### 3D TESTS ####


Ns = [
    3,
    # 5,
    # 8,
]

length_modifiers = [
    # -1,
    0,
    1,
    # 4
]


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_3D_backward_CPU_targets(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, (3, N + modifier), fftshift, isign, "cpu", False)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_3D_backward_CPU_points(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, (3, N + modifier), fftshift, isign, "cpu", True)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_3D_backward_cuda_targets(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, (3, N + modifier), fftshift, isign, "cuda", False)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_3D_backward_cuda_points(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, (3, N + modifier), fftshift, isign, "cuda", True)
