import warnings
import numpy as np
import pytest
import torch

import pytorch_finufft

# devices


def test_t1_mismatch_cuda_non_cuda() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128).to("cuda:0")

    with pytest.raises(ValueError, match="Some tensors are not on the same device"):
        pytorch_finufft.functional.finufft_type1.apply(points, values, (10, 10))


def test_t1_mismatch_cuda_index() -> None:
    points = torch.rand((2, 10), dtype=torch.float64).to("cuda:0")
    values = torch.randn(10, dtype=torch.complex128).to("cuda:1")

    with pytest.raises(ValueError, match="Some tensors are not on the same device"):
        pytorch_finufft.functional.finufft_type1.apply(points, values, (10, 10))


# dtypes


def test_t1_non_complex_values() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.float64)

    with pytest.raises(
        TypeError,
        match="Values must have a dtype of torch.complex64 or torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type1.apply(points, values, (10, 10))


def test_t1_half_complex_values() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        values = torch.randn(10, dtype=torch.complex32)

    with pytest.raises(
        TypeError,
        match="Values must have a dtype of torch.complex64 or torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type1.apply(points, values, (10, 10))


def test_t1_non_real_points() -> None:
    points = torch.rand((2, 10), dtype=torch.complex128)
    values = torch.randn(10, dtype=torch.complex128)

    with pytest.raises(
        TypeError,
        match="Points must have a dtype of torch.float64 as values has "
        "a dtype of torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type1.apply(points, values, (10, 10))


def test_t1_mismatch_precision() -> None:
    points = torch.rand((2, 10), dtype=torch.float32)
    values = torch.randn(10, dtype=torch.complex128)

    with pytest.raises(
        TypeError,
        match="Points must have a dtype of torch.float64 as values has "
        "a dtype of torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type1.apply(points, values, (10, 10))

    points = torch.rand((2, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex64)

    with pytest.raises(
        TypeError,
        match="Points must have a dtype of torch.float32 as values has "
        "a dtype of torch.complex64",
    ):
        pytorch_finufft.functional.finufft_type1.apply(points, values, (10, 10))


# sizes


def test_t1_wrong_length() -> None:
    points = torch.rand(10, dtype=torch.float64)
    values = torch.randn(12, dtype=torch.complex128)

    with pytest.raises(
        ValueError, match="The same number of points and values must be supplied"
    ):
        pytorch_finufft.functional.finufft_type1.apply(points, values, (10, 10))

    points = torch.rand((3, 10), dtype=torch.float64)

    with pytest.raises(
        ValueError, match="The same number of points and values must be supplied"
    ):
        pytorch_finufft.functional.finufft_type1.apply(points, values, (10, 10))


def test_t1_points_4d() -> None:
    points = torch.rand((4, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128)

    with pytest.raises(ValueError, match="Points can be at most 3d, got"):
        pytorch_finufft.functional.finufft_type1.apply(points, values, (10, 10))


def test_t1_too_many_points_dims() -> None:
    points = torch.rand((1, 4, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128)

    with pytest.raises(ValueError, match="The points tensor must be 1d or 2d"):
        pytorch_finufft.functional.finufft_type1.apply(points, values, (10, 10))


def test_t1_wrong_output_dims() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128)

    with pytest.raises(
        ValueError, match="output_shape must be of length 2 for 2d NUFFT"
    ):
        pytorch_finufft.functional.finufft_type1.apply(points, values, (10, 10, 10))


def test_t1_negative_output_dims() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128)

    with pytest.raises(
        ValueError, match="Got output_shape that was not positive integer"
    ):
        pytorch_finufft.functional.finufft_type1.apply(points, values, 0)
    with pytest.raises(
        ValueError, match="Got output_shape that was not positive integer"
    ):
        pytorch_finufft.functional.finufft_type1.apply(points, values, (10, -2))


def test_t1_bad_fftshift_modeord() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128)

    with pytest.raises(
        ValueError,
        match="Conflict between argument fftshift and FINUFFT keyword argument modeord",
    ):
        pytorch_finufft.functional.finufft_type1.apply(
            points, values, (10, 10), None, True, dict(modeord=0)
        )

