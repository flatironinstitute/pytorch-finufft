import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

import pytorch_finufft

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)


######################################################################
# TEST CASES
######################################################################


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


######################################################################
# TYPE 1 TESTS
######################################################################


def check_t1_backward(
    N: int,
    modifier: int,
    fftshift: bool,
    isign: int,
    device: str,
    points_or_values: bool,
) -> None:
    points = torch.rand((2, N), dtype=torch.float64).to(device) * 2 * np.pi
    values = torch.randn(N, dtype=torch.complex128).to(device)

    points.requires_grad = points_or_values
    values.requires_grad = not points_or_values

    inputs = (points, values)

    def func(points, values):
        return pytorch_finufft.functional.finufft_type1.apply(
            points,
            values,
            (N, N + modifier),
            dict(modeord=int(not fftshift), isign=isign),
        )

    assert gradcheck(func, inputs, atol=1e-5 * N)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_backward_CPU_points(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t1_backward(N, modifier, fftshift, isign, "cpu", True)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_backward_CPU_values(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t1_backward(N, modifier, fftshift, isign, "cpu", False)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_backward_cuda_values(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t1_backward(N, modifier, fftshift, isign, "cuda", False)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_backward_cuda_points(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t1_backward(N, modifier, fftshift, isign, "cuda", True)


def check_t2_backward(
    N: int,
    modifier: int,
    fftshift: bool,
    isign: int,
    device: str,
    points_or_targets: bool,
) -> None:
    points = torch.rand((2, N + modifier), dtype=torch.float64).to(device) * 2 * np.pi
    targets = torch.randn(N, N, dtype=torch.complex128).to(device)

    points.requires_grad = points_or_targets
    targets.requires_grad = not points_or_targets

    inputs = (points, targets)

    def func(points, targets):
        return pytorch_finufft.functional.finufft_type2.apply(
            points,
            targets,
            dict(modeord=int(not fftshift), isign=isign),
        )

    assert gradcheck(func, inputs, atol=1e-5 * N)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_backward_CPU_targets(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, modifier, fftshift, isign, "cpu", False)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_backward_CPU_points(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, modifier, fftshift, isign, "cpu", True)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_backward_cuda_targets(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, modifier, fftshift, isign, "cuda", False)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_backward_cuda_points(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t2_backward(N, modifier, fftshift, isign, "cuda", True)
