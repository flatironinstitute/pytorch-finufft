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


def apply_finufft1d1(modifier: int, fftshift: bool, isign: int):
    """Wrappper around finufft1D1.apply(...)"""

    def f(points: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return pytorch_finufft.functional.finufft1D1.apply(
            points,
            values,
            len(values) + modifier,
            None,
            fftshift,
            dict(isign=isign),
        )

    return f


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


######################################################################
# TYPE 1 TESTS
######################################################################

# Case generation
Ns = [
    5,
    8,
    10,
    15,
    16,
    55,
    63,
    100,
    101,
]

length_modifiers = [0, 1, 10]


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_backward_CPU_points(
    N: int, modifier: int, fftshift: bool, isign: int
) -> None:
    """
    Uses gradcheck to test the correctness of the implementation
    of the points gradients for NUFFT type 2 in functional
    """
    points = 3 * np.pi * ((2 * torch.rand(N, dtype=torch.float64)) - 1)
    values = torch.randn(N, dtype=torch.complex128)

    values.requires_grad = False
    points.requires_grad = True

    inputs = (points, values)

    assert gradcheck(
        apply_finufft1d1(modifier, fftshift, isign), inputs, atol=1e-4 * N
    )


# Case generation for the values tests
cases = [torch.tensor([1.0, 2.5, -1.0, -1.5, 1.5], dtype=torch.complex128)]

for n in Ns:
    cases.append(torch.randn(n, dtype=torch.complex128))


@pytest.mark.parametrize("values", cases)
@pytest.mark.parametrize("modifier", length_modifiers)
@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_backward_CPU_values(
    values: torch.Tensor, modifier: int, fftshift: bool, isign: int
) -> None:
    """
    Uses gradcheck to test the correctness of the implementation
    of the values gradients for NUFFT type 1 in functional
    """
    N = len(values)
    points = 2 * np.pi * torch.arange(0, 1, 1 / N, dtype=torch.float64)

    values.requires_grad = True
    points.requires_grad = False

    inputs = (points, values)

    assert gradcheck(apply_finufft1d1(modifier, fftshift, isign), inputs)


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
    # TODO test points.size != targets.size

    targets.requires_grad = False
    points.requires_grad = True

    inputs = (points, targets)

    assert gradcheck(
        apply_finufft1d2(fftshift, isign), inputs
    )
