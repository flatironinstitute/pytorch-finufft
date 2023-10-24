from typing import Tuple, Union

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

import pytorch_finufft

torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)


def check_t1_backward(
    N: int,
    output_shape: Union[int, Tuple[int, ...]],
    fftshift: bool,
    isign: int,
    device: str,
    points_or_values: bool,
) -> None:
    if isinstance(output_shape, int):
        output_shape = (output_shape,)

    dim = len(output_shape)
    points = torch.rand((dim, N), dtype=torch.float64).to(device) * 2 * np.pi
    values = torch.randn(N, dtype=torch.complex128).to(device)

    points.requires_grad = points_or_values
    values.requires_grad = not points_or_values

    inputs = (points, values)

    def func(points, values):
        return pytorch_finufft.functional.finufft_type1(
            points,
            values,
            output_shape,
            modeord=int(not fftshift),
            isign=isign,
        )

    assert gradcheck(func, inputs, eps=1e-8, atol=2e-4 * N)


shapes_and_Ns = [
    (2, 2),
    (2, 51),
    (5, 4),
    (6, 50),
    (101, 10),
    ((2,), 10),
    ((5,), 25),
    ((2, 2), 21),
    ((20, 21), 51),
    ((8, 30), 23),
    ((5, 4, 3), 10),
]


@pytest.mark.parametrize("output_shape, N", shapes_and_Ns)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_backward_CPU_points(output_shape, N, fftshift, isign) -> None:
    check_t1_backward(N, output_shape, fftshift, isign, "cpu", True)


@pytest.mark.parametrize("output_shape, N", shapes_and_Ns)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_backward_CPU_values(output_shape, N, fftshift, isign) -> None:
    check_t1_backward(N, output_shape, fftshift, isign, "cpu", False)


@pytest.mark.parametrize("output_shape, N", shapes_and_Ns)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_backward_cuda_points(output_shape, N, fftshift, isign) -> None:
    check_t1_backward(N, output_shape, fftshift, isign, "cuda", True)


@pytest.mark.parametrize("output_shape, N", shapes_and_Ns)
@pytest.mark.parametrize("fftshift", [False, True])
@pytest.mark.parametrize("isign", [-1, 1])
def test_t1_backward_cuda_values(output_shape, N, fftshift, isign) -> None:
    check_t1_backward(N, output_shape, fftshift, isign, "cuda", False)
