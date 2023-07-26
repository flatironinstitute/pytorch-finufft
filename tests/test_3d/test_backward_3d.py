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


def apply_finufft3d1(
    points_x: torch.Tensor,
    points_y: torch.Tensor,
    points_z: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Wrapper around finufft2D1.apply(...)"""
    return pytorch_finufft.functional.finufft3D1.apply(
        points_x, points_y, points_z, values, len(values)
    )


def apply_finufft3d2(
    points_x: torch.Tensor,
    points_y: torch.Tensor,
    points_z: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Wrapper around finufft2D1.apply(...)"""
    return pytorch_finufft.functional.finufft3D2.apply(
        points_x, points_y, points_z, targets
    )


######################################################################
# TEST CASES
######################################################################


Ns = [
    3,
    5,
    8,
    10,
    15,
    16,
    17,
    20,
]


######################################################################
# TYPE 1 TESTS
######################################################################


@pytest.mark.parametrize("N", Ns)
def test_t1_backward_CPU_values(N: int) -> None:
    """
    Uses gradcheck to test the correctness of the implementation
    of the derivative in values for 2d NUFFT type 1
    """
    points_x = 2 * np.pi * torch.arange(0, 1, 1 / N)
    points_y = 2 * np.pi * torch.arange(0, 1, 1 / N)
    points_z = 2 * np.pi * torch.arange(0, 1, 1 / N)
    values = torch.rand(N, dtype=torch.complex128)

    points_x.requires_grad = False
    points_y.requires_grad = False
    points_z.requires_grad = False
    values.requires_grad = True

    inputs = (points_x, points_y, points_z, values)

    assert gradcheck(apply_finufft3d1, inputs)


@pytest.mark.parametrize("N", Ns)
def test_t1_backward_CPU_points_x(N: int) -> None:
    """
    Uses gradcheck to test the correctness of the implementation
    of the derivative in points_x for 2d NUFFT type 1
    """

    points_x = 3 * np.pi * (torch.rand(N) - (torch.ones(N) / 2))
    points_y = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    points_z = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    values = torch.randn(N)

    points_x.requires_grad = True
    points_y.requires_grad = False
    points_z.requires_grad = False
    values.requires_grad = False

    inputs = (points_x, points_y, points_z, values)

    assert gradcheck(apply_finufft3d1, inputs)


@pytest.mark.parametrize("N", Ns)
def test_t1_backward_CPU_points_y(N: int) -> None:
    """
    Uses gradcheck to test the correctness of the implementation
    of the derivative in points_y for 2d NUFFT type 1
    """

    points_x = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    points_y = 3 * np.pi * (torch.rand(N) - (torch.ones(N) / 2))
    points_z = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    values = torch.randn(N)

    points_x.requires_grad = False
    points_y.requires_grad = True
    points_z.requires_grad = False
    values.requires_grad = False

    inputs = (points_x, points_y, points_z, values)

    assert gradcheck(apply_finufft3d1, inputs)


@pytest.mark.parametrize("N", Ns)
def test_t1_backward_CPU_points_z(N: int) -> None:
    """
    Uses gradcheck to test the correctness of the implementation
    of the derivative in points_y for 2d NUFFT type 1
    """

    points_x = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    points_y = 3 * np.pi * (torch.rand(N) - (torch.ones(N) / 2))
    points_z = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    values = torch.randn(N)

    points_x.requires_grad = False
    points_y.requires_grad = False
    points_z.requires_grad = True
    values.requires_grad = False

    inputs = (points_x, points_y, points_z, values)

    assert gradcheck(apply_finufft3d1, inputs)


######################################################################
# TYPE 2 TESTS
######################################################################


@pytest.mark.parametrize("N", Ns)
def test_t2_backward_CPU_targets(N: int) -> None:
    """
    Uses gradcheck to test the correctness of the implemntation
    of the derivative in targets for 2d NUFFT type 2
    """

    points_x = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    points_y = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    targets = torch.randn(2 * N)

    points_x.requires_grad = False
    points_y.requires_grad = False
    targets.requires_grad = True

    inputs = (points_x, points_y, targets)

    assert gradcheck(apply_finufft2d2, inputs)


@pytest.mark.parametrize("N", Ns)
def test_t2_backward_CPU_points_x(N: int) -> None:
    """
    Uses gradcheck to test the correctness of the implemntation
    of the derivative in targets for 2d NUFFT type 2
    """

    points_x = 3 * np.pi * (torch.rand(N) - (torch.ones(N) / 2))
    points_y = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    targets = torch.randn(2 * N)

    points_x.requires_grad = True
    points_y.requires_grad = False
    targets.requires_grad = False

    inputs = (points_x, points_y, targets)

    assert gradcheck(apply_finufft2d2, inputs)


@pytest.mark.parametrize("N", Ns)
def test_t2_backward_CPU_points_y(N: int) -> None:
    """
    Uses gradcheck to test the correctness of the implemntation
    of the derivative in targets for 2d NUFFT type 2
    """

    points_x = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)
    points_y = 3 * np.pi * (torch.rand(N) - (torch.ones(N) / 2))
    targets = torch.randn(2 * N)

    points_x.requires_grad = False
    points_y.requires_grad = True
    targets.requires_grad = False

    inputs = (points_x, points_y, targets)

    assert gradcheck(apply_finufft2d2, inputs)
