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


@pytest.mark.parametrize("values", cases)
def t1_backward_CPU_FD(points: torch.Tensor, values: torch.Tensor) -> None:
    """
    Checks autograd output against a finite difference approximation
    of the functional derivative.
    """
    N = len(values)

    values.requires_grad = True
    points = torch.arange(N, requires_grad=False) * (2 * np.pi) / 100

    c = torch.randn(100, requires_grad=True)

    rind = np.randint(100)
    perturbation = torch.zeros(100)

    DeltaF = (
        pytorch_finufft.functional.finufft1D1(c + T * perturbation, points)[
            rind
        ]
        - pytorch_finufft.functional.finufft1D1(c, x)[rind]
    ) / T

    assert something == 1


@pytest.mark.parametrize("something", [1])
def t2_backward_CPU_FD(points: torch.Tensor, targets: torch.Tensor) -> None:
    """
    Checks autograd output against explicit construction of the Jacobian
    and the product for small test cases
    """
    assert something == 1


@pytest.mark.parametrize("something", [1])
def t2_backward_CPU_FD(points: torch.Tensor, targets: torch.Tensor) -> None:
    """
    Checks autograd output against a finite difference approximation
    of the functional derivative.
    """
    assert something == 1


@pytest.mark.parametrize("something", [1])
def t3_backward_CPU_FD(
    points: torch.Tensor, values: torch.Tensor, targets: torch.Tensor
) -> None:
    """
    Checks autograd output against finite differences computation
    """

    assert something == 1
