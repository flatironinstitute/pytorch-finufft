import numpy as np
import pytest
import scipy
import torch
from numpy.random import standard_normal

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
    1000,
    1001,
    2500,
    3750,
    5000,
    5001,
    6250,
    7500,
    8750,
    10000,
]
cases = [torch.tensor([1.0, 2.5, -1.0, -1.5, 1.5], dtype=torch.complex128)]
for n in Ns:
    cases.append(
        torch.randn(n, dtype=torch.float64)
        + 1j * torch.randn(n, dtype=torch.float64)
    )
    cases.append(
        torch.randn(n, dtype=torch.float32)
        + 1j * torch.randn(n, dtype=torch.float32)
    )


######################################################################
# TYPE 1 TESTS
######################################################################


@pytest.mark.parametrize("values", cases)
def t1_backward_CPU_values(values: torch.Tensor) -> None:
    """
    Checks autograd output against a finite difference approximation
    of the functional derivative.
    """
    N = len(values)

    values.requires_grad = True
    points = torch.arange(N, requires_grad=False) * (2 * np.pi) / 100

    rind = np.randint(100)
    w = torch.zeros(100)
    w[rind] = 1
    V = torch.randn(N)

    # Frechet test

    out = pytorch_finufft.functional.finufft1D1.apply(points, values)
    JAC_w_F = torch.abs(out).flatten().dot(V)

    assert values.grad is None
    JAC_w_F.backward()
    assert values.grad is not None

    assert torch.dot(w, values.grad) - (
        torch.abs(pytorch_finufft.functional.finufft1D1(points, values + T * w))
        .flatten()
        .dot(V)
        - torch.abs(pytorch_finufft.functional.finufft1D1(points, values))
        .flatten()
        .dot(V)
    ) / T == pytest.approx(0, abs=1e-05)

    # GD test
    out = pytorch_finufft.functional.finufft1D1.apply(points, values)
    assert out.dtype is torch.complex64

    norm_out = torch.norm(out)
    assert values.grad is None

    norm_out.backward()
    assert values.grad is not None

    d = values - (1e-6 * values.grad)
    grad_desc = pytorch_finufft.functional.finufft1D1.apply(points, d)

    assert torch.norm(grad_desc) < norm_out


@pytest.mark.parametrize("points", [1])
def t1_backward_CPU_points(points: torch.Tensor, targets: torch.Tensor) -> None:
    """
    Checks autograd output against explicit construction of the Jacobian
    and the product for small test cases
    """
    assert points == 1


######################################################################
# TYPE 2 TESTS
######################################################################


@pytest.mark.parametrize("something", [1])
def t2_backward_CPU_values(points: torch.Tensor, targets: torch.Tensor) -> None:
    """
    Checks autograd output against explicit construction of the Jacobian
    and the product for small test cases
    """
    assert something == 1


@pytest.mark.parametrize("something", [1])
def t2_backward_CPU_points(points: torch.Tensor, targets: torch.Tensor) -> None:
    """
    Checks autograd output against a finite difference approximation
    of the functional derivative.
    """
    assert something == 1


######################################################################
# TYPE 1 TESTS
######################################################################


@pytest.mark.parametrize("something", [1])
def t3_backward_CPU_FD(
    points: torch.Tensor, values: torch.Tensor, targets: torch.Tensor
) -> None:
    """
    Checks autograd output against finite differences computation
    """

    assert something == 1
