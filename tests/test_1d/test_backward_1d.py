import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

import pytorch_finufft

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


# Case generation
Ns = [
    5,
    8,
    10,
    15,
    16,
    63,
    100,
    101,
]

length_modifiers = [-1, 0, 1, 4]


def check_t1_backward(
    N: int,
    modifier: int,
    fftshift: bool,
    isign: int,
    device: str,
    points_or_values: bool,
) -> None:
    # TODO: we should also test shape (N,) for points - don't want combinatorics for now
    points = torch.rand((1, N), dtype=torch.float64).to(device) * 2 * np.pi
    values = torch.randn(N, dtype=torch.complex128).to(device)

    points.requires_grad = points_or_values
    values.requires_grad = not points_or_values

    inputs = (points, values)

    def func(points, values):
        return pytorch_finufft.functional.finufft_type1.apply(
            points,
            values,
            (N + modifier,),
            None,
            dict(modeord=int(not fftshift), isign=isign),
        )

    assert gradcheck(func, inputs, eps=1e-8, atol=1e-5 * N)


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
def test_t1_backward_CPU_points(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    check_t1_backward(N, modifier, fftshift, isign, "cpu", True)


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


######################################################################
# TYPE 2 TESTS
######################################################################


def apply_finufft1d2(fftshift: bool, isign: int):
    """Wrapper around finufft1D2.apply(...)"""

    def f(points: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return pytorch_finufft.functional.finufft1D2.apply(
            points,
            targets,
            None,
            fftshift,
            dict(isign=isign),
        )

    return f


"""
NOTE: A few of the below do NOT pass due to strict tolerance
"""


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_backward_CPU_targets(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    """
    Uses gradcheck to test the correctness of the implementation of
    targets gradients for NUFFT type 2 in functional.
    """
    points = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    targets = torch.randn(N + modifier, dtype=torch.complex128)

    targets.requires_grad = True
    points.requires_grad = False

    inputs = (points, targets)

    assert gradcheck(apply_finufft1d2(fftshift, isign), inputs)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_backward_CPU_points(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    """
    Uses gradcheck to test the correctness of the implementation of
    targets gradients for NUFFT type 2 in functional.
    """
    points = 3 * np.pi * ((2 * torch.rand(N, dtype=torch.float64)) - 1)
    targets = torch.randn(N + modifier, dtype=torch.complex128)

    targets.requires_grad = False
    points.requires_grad = True

    inputs = (points, targets)

    assert gradcheck(apply_finufft1d2(fftshift, isign), inputs, eps=1e-8, atol=1e-5 * N)
