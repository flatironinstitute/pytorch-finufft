import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

import pytorch_finufft

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

T = 1e-5

# Case generation
Ns = [
    10,
    15,
    100,
    101,
]
cases = [torch.tensor([1.0, 2.5, -1.0, -1.5, 1.5], dtype=torch.complex128)]
for n in Ns:
    cases.append(torch.randn(n, dtype=torch.complex128))

######################################################################
# APPLY WRAPPERS
######################################################################


def apply_finufft1d1(
    points: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Wrappper around finufft1D1.apply(...)"""
    return pytorch_finufft.functional.finufft1D1.apply(
        points, values, len(values)
    )


def apply_finufft1d2(
    points: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Wrapper around finufft1D2.apply(...)"""
    return pytorch_finufft.functional.finufft1D2.apply(points, targets)


######################################################################
# TYPE 1 TESTS
######################################################################


@pytest.mark.parametrize("values", cases)
def test_t1_backward_CPU_values(values: torch.Tensor) -> None:
    """
    Uses gradcheck to test the correctness of the implementation
    of the values gradients for NUFFT type 2 in functional
    """
    N = len(values)
    points = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)

    values.requires_grad = True
    points.requires_grad = False

    inputs = (points, values)

    gradcheck(apply_finufft1d1, inputs)


@pytest.mark.parametrize("values", cases)
def test_t1_backward_CPU_points(values: torch.Tensor) -> None:
    """
    Uses gradcheck to test the correctness of the implementation
    of the points gradients for NUFFT type 2 in functional
    """
    N = len(values)
    points = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)

    values.requires_grad = False
    points.requires_grad = True

    inputs = (points, values)

    assert gradcheck(apply_finufft1d1, inputs)


######################################################################
# TYPE 2 TESTS
######################################################################


@pytest.mark.parametrize("targets", cases)
def test_t2_forward_CPU_targets(targets: torch.Tensor) -> None:
    """
    Uses gradcheck to test the correctness of the implementation of
    targets gradients for NUFFT type 2 in functional.
    """
    pass


@pytest.mark.parametrize("targets", cases)
def test_t2_forward_CPU_points(targets: torch.Tensor) -> None:
    """
    Uses gradcheck to test the correctness of the implementation of
    targets gradients for NUFFT type 2 in functional.
    """
    pass
