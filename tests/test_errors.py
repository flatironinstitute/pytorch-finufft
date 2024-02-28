import warnings

import numpy as np
import pytest
import torch

import pytorch_finufft

torch.manual_seed(1234)
# devices


def test_t1_mismatch_device_cuda_cpu() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128).to("cuda:0")

    with pytest.raises(ValueError, match="Some tensors are not on the same device"):
        pytorch_finufft.functional.finufft_type1(points, values, (10, 10))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="require multiple GPUs")
def test_t1_mismatch_cuda_index() -> None:
    points = torch.rand((2, 10), dtype=torch.float64).to("cuda:0")
    values = torch.randn(10, dtype=torch.complex128).to("cuda:1")

    with pytest.raises(ValueError, match="Some tensors are not on the same device"):
        pytorch_finufft.functional.finufft_type1(points, values, (10, 10))


def test_t2_mismatch_device_cuda_cpu() -> None:
    g = np.mgrid[:10, :10] * 2 * np.pi / 10
    points = torch.from_numpy(g.reshape(2, -1))
    targets = torch.randn(*g[0].shape, dtype=torch.complex128).to("cuda:0")

    with pytest.raises(ValueError, match="Some tensors are not on the same device"):
        pytorch_finufft.functional.finufft_type2(points, targets)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="require multiple GPUs")
def test_t2_mismatch_cuda_index() -> None:
    g = np.mgrid[:10, :10] * 2 * np.pi / 10
    points = torch.from_numpy(g.reshape(2, -1)).to("cuda:0")
    targets = torch.randn(*g[0].shape, dtype=torch.complex128).to("cuda:1")

    with pytest.raises(ValueError, match="Some tensors are not on the same device"):
        pytorch_finufft.functional.finufft_type2(points, targets)


def test_t3_mismatch_device_cuda_cpu() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128)
    targets = torch.rand((2, 10), dtype=torch.float64)

    with pytest.raises(ValueError, match="Some tensors are not on the same device"):
        pytorch_finufft.functional.finufft_type3(points.to("cuda:0"), values, targets)

    with pytest.raises(ValueError, match="Some tensors are not on the same device"):
        pytorch_finufft.functional.finufft_type3(points, values.to("cuda:0"), targets)

    with pytest.raises(ValueError, match="Some tensors are not on the same device"):
        pytorch_finufft.functional.finufft_type3(points, values, targets.to("cuda:0"))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="require multiple GPUs")
def test_t3_mismatch_cuda_index() -> None:
    points = torch.rand((2, 10), dtype=torch.float64).to("cuda:0")
    values = torch.randn(10, dtype=torch.complex128).to("cuda:0")
    targets = torch.rand((2, 10), dtype=torch.float64).to("cuda:0")

    with pytest.raises(ValueError, match="Some tensors are not on the same device"):
        pytorch_finufft.functional.finufft_type3(points.to("cuda:1"), values, targets)

    with pytest.raises(ValueError, match="Some tensors are not on the same device"):
        pytorch_finufft.functional.finufft_type3(points, values.to("cuda:1"), targets)

    with pytest.raises(ValueError, match="Some tensors are not on the same device"):
        pytorch_finufft.functional.finufft_type3(points, values, targets.to("cuda:1"))


# dtypes


def test_t1_non_complex_values() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.float64)

    with pytest.raises(
        TypeError,
        match="Values must have a dtype of torch.complex64 or torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type1(points, values, (10, 10))


def test_t1_half_complex_values() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        values = torch.randn(10, dtype=torch.complex32)

    with pytest.raises(
        TypeError,
        match="Values must have a dtype of torch.complex64 or torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type1(points, values, (10, 10))


def test_t1_non_real_points() -> None:
    points = torch.rand((2, 10), dtype=torch.complex128)
    values = torch.randn(10, dtype=torch.complex128)

    with pytest.raises(
        TypeError,
        match="Points must have a dtype of torch.float64 as values has "
        "a dtype of torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type1(points, values, (10, 10))


def test_t1_mismatch_precision() -> None:
    points = torch.rand((2, 10), dtype=torch.float32)
    values = torch.randn(10, dtype=torch.complex128)

    with pytest.raises(
        TypeError,
        match="Points must have a dtype of torch.float64 as values has "
        "a dtype of torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type1(points, values, (10, 10))

    points = torch.rand((2, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex64)

    with pytest.raises(
        TypeError,
        match="Points must have a dtype of torch.float32 as values has "
        "a dtype of torch.complex64",
    ):
        pytorch_finufft.functional.finufft_type1(points, values, (10, 10))


def test_t2_non_complex_targets() -> None:
    g = np.mgrid[:10, :10] * 2 * np.pi / 10
    points = torch.from_numpy(g.reshape(2, -1))
    targets = torch.randn(*g[0].shape, dtype=torch.float64)

    with pytest.raises(
        TypeError,
        match="Targets must have a dtype of torch.complex64 or torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type2(points, targets)


def test_t2_half_complex_targets() -> None:
    g = np.mgrid[:10, :10] * 2 * np.pi / 10
    points = torch.from_numpy(g.reshape(2, -1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        targets = torch.randn(*g[0].shape, dtype=torch.complex32)

    with pytest.raises(
        TypeError,
        match="Targets must have a dtype of torch.complex64 or torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type2(points, targets)


def test_t2_non_real_points() -> None:
    g = np.mgrid[:10, :10] * 2 * np.pi / 10
    points = torch.from_numpy(g.reshape(2, -1)).to(torch.complex128)
    targets = torch.randn(*g[0].shape, dtype=torch.complex128)

    with pytest.raises(
        TypeError,
        match="Points must have a dtype of torch.float64 as targets has "
        "a dtype of torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type2(points, targets)


def test_t2_mismatch_precision() -> None:
    g = np.mgrid[:10, :10] * 2 * np.pi / 10
    points = torch.from_numpy(g.reshape(2, -1)).to(torch.float32)
    targets = torch.randn(*g[0].shape, dtype=torch.complex128)

    with pytest.raises(
        TypeError,
        match="Points must have a dtype of torch.float64 as targets has "
        "a dtype of torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type2(points, targets)

    points = points.to(torch.float64)
    targets = targets.to(torch.complex64)

    with pytest.raises(
        TypeError,
        match="Points must have a dtype of torch.float32 as targets has "
        "a dtype of torch.complex64",
    ):
        pytorch_finufft.functional.finufft_type2(points, targets)


def test_t3_non_complex_targets() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.float64)
    targets = torch.rand((2, 10), dtype=torch.float64)

    with pytest.raises(
        TypeError,
        match="Strengths must have a dtype of torch.complex64 or torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type3(points, values, targets)


def test_t3_half_complex_targets() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)
    targets = torch.rand((2, 10), dtype=torch.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        values = torch.randn(10, dtype=torch.complex32)

    with pytest.raises(
        TypeError,
        match="Strengths must have a dtype of torch.complex64 or torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type3(points, values, targets)


def test_t3_non_real_points() -> None:
    points = torch.rand((2, 10), dtype=torch.complex128)
    values = torch.randn(10, dtype=torch.complex128)
    targets = torch.rand((2, 10), dtype=torch.float64)

    with pytest.raises(
        TypeError,
        match="Points must have a dtype of torch.float64 as strengths has "
        "a dtype of torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type3(points, values, targets)


def test_t3_non_real_targets() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128)
    targets = torch.rand((2, 10), dtype=torch.complex128)

    with pytest.raises(
        TypeError,
        match="Targets must have a dtype of torch.float64 as strengths has "
        "a dtype of torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type3(points, values, targets)


def test_t3_mismatch_precision() -> None:
    points = torch.rand((2, 10), dtype=torch.float32)
    values = torch.randn(10, dtype=torch.complex128)
    targets = torch.rand((2, 10), dtype=torch.float64)

    with pytest.raises(
        TypeError,
        match="Points must have a dtype of torch.float64 as strengths has "
        "a dtype of torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type3(points, values, targets)

    points = points.to(torch.float64)
    targets = targets.to(torch.float32)

    with pytest.raises(
        TypeError,
        match="Targets must have a dtype of torch.float64 as strengths has "
        "a dtype of torch.complex128",
    ):
        pytorch_finufft.functional.finufft_type3(points, values, targets)

    values = values.to(torch.complex64)

    with pytest.raises(
        TypeError,
        match="Points must have a dtype of torch.float32 as strengths has "
        "a dtype of torch.complex64",
    ):
        pytorch_finufft.functional.finufft_type3(points, values, targets)

    points = points.to(torch.float32)
    targets = targets.to(torch.float64)

    with pytest.raises(
        TypeError,
        match="Targets must have a dtype of torch.float32 as strengths has "
        "a dtype of torch.complex64",
    ):
        pytorch_finufft.functional.finufft_type3(points, values, targets)


# sizes


def test_t1_wrong_length() -> None:
    points = torch.rand(10, dtype=torch.float64)
    values = torch.randn(12, dtype=torch.complex128)

    with pytest.raises(
        ValueError, match="The same number of points and values must be supplied"
    ):
        pytorch_finufft.functional.finufft_type1(points, values, (10,))

    points = torch.rand((3, 10), dtype=torch.float64)

    with pytest.raises(
        ValueError, match="The same number of points and values must be supplied"
    ):
        pytorch_finufft.functional.finufft_type1(points, values, (10,))


def test_t1_points_4d() -> None:
    points = torch.rand((4, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128)

    with pytest.raises(ValueError, match="Points can be at most 3d, got"):
        pytorch_finufft.functional.finufft_type1(points, values, (10, 10))


def test_t1_too_many_points_dims() -> None:
    points = torch.rand((1, 4, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128)

    with pytest.raises(ValueError, match="The points tensor must be 1d or 2d"):
        pytorch_finufft.functional.finufft_type1(points, values, (10, 10))


def test_t1_wrong_output_dims() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128)

    with pytest.raises(
        ValueError, match="output_shape must be of length 2 for 2d NUFFT"
    ):
        pytorch_finufft.functional.finufft_type1(points, values, (10, 10, 10))

    with pytest.raises(
        ValueError, match="output_shape must be of length 2 for 2d NUFFT"
    ):
        pytorch_finufft.functional.finufft_type1(points, values, (10,))

    with pytest.raises(
        ValueError, match="output_shape must be a tuple of length 2 for 2d NUFFT"
    ):
        pytorch_finufft.functional.finufft_type1(points, values, 10)


def test_t1_negative_output_dims() -> None:
    points = torch.rand(10, dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128)

    with pytest.raises(
        ValueError, match="Got output_shape that was not positive integer"
    ):
        pytorch_finufft.functional.finufft_type1(points, values, 0)

    points = torch.rand((2, 10), dtype=torch.float64)
    with pytest.raises(
        ValueError, match="Got output_shape that was not positive integer"
    ):
        pytorch_finufft.functional.finufft_type1(points, values, (10, -2))


def test_t2_points_4d() -> None:
    g = np.mgrid[:10, :10, :10, :10] * 2 * np.pi / 10
    points = torch.from_numpy(g.reshape(4, -1)).to(torch.float64)
    targets = torch.randn(*g[0].shape, dtype=torch.complex128)

    with pytest.raises(ValueError, match="Points can be at most 3d, got"):
        pytorch_finufft.functional.finufft_type2(points, targets)


def test_t2_too_many_points_dims() -> None:
    g = np.mgrid[:10, :10] * 2 * np.pi / 10
    points = torch.from_numpy(g.reshape(1, 2, -1)).to(torch.float64)
    targets = torch.randn(*g[0].shape, dtype=torch.complex128)

    with pytest.raises(ValueError, match="The points tensor must be 1d or 2d"):
        pytorch_finufft.functional.finufft_type2(points, targets)


def test_t2_mismatch_dims() -> None:
    g = np.mgrid[:10, :10, :10] * 2 * np.pi / 10
    points = torch.from_numpy(g.reshape(3, -1)).to(torch.float64)
    targets = torch.randn(*g[0].shape[:-1], dtype=torch.complex128)

    with pytest.raises(
        ValueError, match="For type 2 3d FINUFFT, targets must be at least a 3d tensor"
    ):
        pytorch_finufft.functional.finufft_type2(points, targets)


def test_t3_points_targets_4d() -> None:
    points = torch.rand((4, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128)
    targets = torch.rand((4, 10), dtype=torch.float64)

    with pytest.raises(ValueError, match="Points and targets can be at most 3d, got"):
        pytorch_finufft.functional.finufft_type3(points, values, targets)


def test_t3_points_targets_mismatch_dims() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128)
    targets = torch.rand((3, 10), dtype=torch.float64)

    with pytest.raises(
        ValueError, match="Points and targets must be of the same dimension"
    ):
        pytorch_finufft.functional.finufft_type3(points, values, targets)


def test_t3_too_many_dims() -> None:
    points = torch.rand((1, 4, 10), dtype=torch.float64)
    values = torch.randn(10, dtype=torch.complex128)
    targets = torch.rand((1, 4, 10), dtype=torch.float64)

    with pytest.raises(ValueError, match="The points tensor must be 1d or 2d"):
        pytorch_finufft.functional.finufft_type3(points, values, targets)

    points = torch.rand((2, 10), dtype=torch.float64)

    with pytest.raises(ValueError, match="The targets tensor must be 1d or 2d"):
        pytorch_finufft.functional.finufft_type3(points, values, targets)


def test_t3_mismatch_dims() -> None:
    points = torch.rand((2, 10), dtype=torch.float64)
    values = torch.randn(11, dtype=torch.complex128)
    targets = torch.rand((2, 12), dtype=torch.float64)

    with pytest.raises(
        ValueError, match="The same number of points and strengths must be supplied"
    ):
        pytorch_finufft.functional.finufft_type3(points, values, targets)


# dependencies
def test_finufft_not_installed():
    if not pytorch_finufft.functional.CUFINUFFT_AVAIL:
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        points = torch.rand(10, dtype=torch.float64).to("cuda")
        values = torch.randn(10, dtype=torch.complex128).to("cuda")

        with pytest.raises(RuntimeError, match="cufinufft failed to import"):
            pytorch_finufft.functional.finufft_type1(points, values, 10)

    elif not pytorch_finufft.functional.FINUFFT_AVAIL:
        points = torch.rand(10, dtype=torch.float64).to("cpu")
        values = torch.randn(10, dtype=torch.complex128).to("cpu")

        with pytest.raises(RuntimeError, match="finufft failed to import"):
            pytorch_finufft.functional.finufft_type1(points, values, 10)
