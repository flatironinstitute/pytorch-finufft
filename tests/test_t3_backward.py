from typing import Tuple, Union

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

import pytorch_finufft

torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)


def check_t3_backward(
    input_shape: Union[int, Tuple[int, ...]],
    output_shape: Union[int, Tuple[int, ...]],
    fftshift: bool,
    isign: int,
    device: str,
) -> None:
    if isinstance(output_shape, int):
        output_shape = (output_shape,)

    dim = len(output_shape)
    points = (
        torch.rand((dim, np.prod(input_shape)), dtype=torch.float64).to(device)
        * 2
        * np.pi
    )
    strengths = torch.randn(np.prod(input_shape), dtype=torch.complex128).to(device)
    targets = torch.rand((dim, np.prod(output_shape)), dtype=torch.float64).to(device)
    inputs = (points, strengths, targets)

    points.requires_grad = True
    strengths.requires_grad = True
    targets.requires_grad = True

    def func(points, strengths, targets):
        return pytorch_finufft.functional.finufft_type3(
            points,
            strengths,
            targets,
            modeord=int(not fftshift),
            isign=isign,
            eps=1e-12,
        )

    atol = max(
        2e-4 * np.prod(points.shape),
        3e-4 * np.prod(targets.shape) ** (1.0 / dim),
    )
    assert gradcheck(
        func,
        inputs,
        eps=1e-8,
        atol=atol,
        nondet_tol=1e-8,
    )


inshapes_and_outshapes = [
    ((2,), (2,)),
    ((2,), (51,)),
    ((5,), (4,)),
    ((6,), (50,)),
    ((101,), (10,)),
    ((2,), (10,)),
    ((5,), (25,)),
    ((2, 2), (21, 21)),
    ((20, 21), (51, 52)),
    ((8, 30), (23, 10)),
    ((5, 4, 3), (10, 5, 3)),
]


@pytest.mark.parametrize("input_shape, output_shape", inshapes_and_outshapes)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t3_backward_CPU(input_shape, output_shape, fftshift, isign) -> None:
    check_t3_backward(input_shape, output_shape, fftshift, isign, "cpu")


# @pytest.mark.parametrize("input_shape, output_shape", inshapes_and_outshapes)
# @pytest.mark.parametrize("fftshift", [False, True])
# @pytest.mark.parametrize("isign", [-1, 1])
# def test_t3_backward_cuda(input_shape, output_shape, fftshift, isign) -> None:
#    check_t3_backward(input_shape, output_shape, fftshift, isign, "cuda")
