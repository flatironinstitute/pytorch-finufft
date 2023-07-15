import numpy as np
import pytest
import torch

import pytorch_finufft

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

T = 1e-5

# Case generation
Ns = [
    10,
    #15,
    #100,
    #101,
    #1000,
    #1001,
    #2500,
    #3750,
    #5000,
    #5001,
    #6250,
    #7500,
    #8750,
    #10000,
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
def test_t1_backward_CPU_values(values: torch.Tensor) -> None:
    """
    Checks autograd output against a finite difference approximation
    of the functional derivative.
    """

    N = len(values)
    values.requires_grad = True

    data_type = (
        torch.float64 if values.dtype is torch.complex128 else torch.float32
    )
    points = torch.arange(N, dtype=data_type, requires_grad=False) * (2 * np.pi) / N

    rind = np.random.randint(N)
    w = torch.zeros(N, dtype=values.dtype)
    w[rind] = 1+0j
    V = torch.randn(N, dtype=data_type)

    # Frechet test

    out = pytorch_finufft.functional.finufft1D1.apply(points, values, N)
    assert out.dtype is values.dtype
    JAC_w_F = torch.abs(out).flatten().dot(V)

    assert values.grad is None
    JAC_w_F.backward()
    assert values.grad is not None

    print("VALUES")
    print(values.grad)
    print(w)

    # HERE:
    assert torch.dot(w, values.grad) - (
        torch.abs(pytorch_finufft.functional.finufft1D1.apply(points, values + T * w, N))
        .flatten()
        .dot(V)
        - torch.abs(pytorch_finufft.functional.finufft1D1.apply(points, values, N))
        .flatten()
        .dot(V)
    ) / T == pytest.approx(0, abs=1e-05)

    # Gradient descent test
    out = pytorch_finufft.functional.finufft1D1.apply(points, values, N)
    assert out.dtype is torch.complex64

    norm_out = torch.norm(out)
    assert values.grad is None

    norm_out.backward()
    assert values.grad is not None

    d = values - (1e-6 * values.grad)
    grad_desc = pytorch_finufft.functional.finufft1D1.apply(points, d)

    assert torch.norm(grad_desc) < norm_out
