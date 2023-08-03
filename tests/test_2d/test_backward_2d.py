import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

import pytorch_finufft

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

######################################################################
# APPLY WRAPPERS
######################################################################


def apply_finufft2d1(modifier: int, fftshift: bool, isign: int):
    """Wrapper around finufft2D1.apply(...)"""

    def f(
        points_x: torch.Tensor, points_y: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        return pytorch_finufft.functional.finufft2D1.apply(
            points_x,
            points_y,
            values,
            len(values) + modifier,
            None,
            fftshift,
            dict(isign=isign),
        )

    return f


def apply_finufft2d2(fftshift: bool, isign: int):
    """Wrapper around finufft2D1.apply(...)"""

    def f(
        points_x: torch.Tensor, points_y: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return pytorch_finufft.functional.finufft2D2.apply(
            points_x, points_y, targets, None, fftshift, dict(isign=isign)
        )

    return f


######################################################################
# TEST CASES
######################################################################


Ns = [
    5,
    8,
    10,
    15,
    16,
]

length_modifiers = [-1, 0, 1, 4]


######################################################################
# TYPE 1 TESTS
######################################################################


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_backward_CPU_values(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    """
    Uses gradcheck to test the correctness of the implementation
    of the derivative in values for 2d NUFFT type 1
    """
    # TODO -- have it test also over uneven points_x and points_y
    points_x = 2 * np.pi * torch.arange(0, 1, 1 / N)
    points_y = 2 * np.pi * torch.arange(0, 1, 1 / N)
    values = torch.randn(N, dtype=torch.complex128)

    points_x.requires_grad = False
    points_y.requires_grad = False
    values.requires_grad = True

    inputs = (points_x, points_y, values)

    assert gradcheck(apply_finufft2d1(modifier, fftshift, isign), inputs)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_backward_CPU_points_x(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    """
    Uses gradcheck to test the correctness of the implementation
    of the derivative in points_x for 2d NUFFT type 1
    """

    # TODO -- have it test also over uneven points_x and points_y
    points_x = 3 * np.pi * (torch.rand(N) - (torch.ones(N) / 2))
    points_y = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    values = torch.randn(N, dtype=torch.complex128)

    points_x.requires_grad = True
    points_y.requires_grad = False
    values.requires_grad = False

    inputs = (points_x, points_y, values)

    assert gradcheck(apply_finufft2d1(modifier, fftshift, isign), inputs)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_backward_CPU_points_y(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    """
    Uses gradcheck to test the correctness of the implementation
    of the derivative in points_y for 2d NUFFT type 1
    """

    # TODO -- have it test also over uneven points_x and points_y
    points_x = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    points_y = 3 * np.pi * (torch.rand(N) - (torch.ones(N) / 2))
    values = torch.randn(N, dtype=torch.complex128)

    points_x.requires_grad = False
    points_y.requires_grad = True
    values.requires_grad = False

    inputs = (points_x, points_y, values)

    assert gradcheck(apply_finufft2d1(modifier, fftshift, isign), inputs)


######################################################################
# TYPE 2 TESTS
######################################################################


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_backward_CPU_targets(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    """
    Uses gradcheck to test the correctness of the implemntation
    of the derivative in targets for 2d NUFFT type 2
    """

    # TODO -- need to make sure the points are uneven and varied sufficiently
    points_x = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    points_y = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    targets = torch.randn((N, N), dtype=torch.complex128)

    points_x.requires_grad = False
    points_y.requires_grad = False
    targets.requires_grad = True

    inputs = (points_x, points_y, targets)

    assert gradcheck(apply_finufft2d2(fftshift, isign), inputs)

    # TODO -- have it test also over uneven points_x and points_y


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_backward_CPU_points_x(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    """
    Uses gradcheck to test the correctness of the implemntation
    of the derivative in targets for 2d NUFFT type 2
    """

    # TODO -- need to make sure the points are uneven and varied sufficiently

    points_x = 3 * np.pi * (torch.rand(N) - (torch.ones(N) / 2))
    points_y = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    targets = torch.randn((N, N), dtype=torch.complex128)

    points_x.requires_grad = True
    points_y.requires_grad = False
    targets.requires_grad = False

    inputs = (points_x, points_y, targets)

    assert gradcheck(apply_finufft2d2(fftshift, isign), inputs)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t2_backward_CPU_points_y(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    """
    Uses gradcheck to test the correctness of the implemntation
    of the derivative in targets for 2d NUFFT type 2
    """

    points_x = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    points_y = 3 * np.pi * (torch.rand(N) - (torch.ones(N) / 2))
    targets = torch.randn((N, N), dtype=torch.complex128)

    points_x.requires_grad = False
    points_y.requires_grad = True
    targets.requires_grad = False

    inputs = (points_x, points_y, targets)

    assert gradcheck(apply_finufft2d2(fftshift, isign), inputs)
