from typing import Tuple, Union

import torch


def check_devices(*tensors: torch.Tensor) -> None:
    """
    Checks that all tensors are on the same device
    """

    device = tensors[0].device
    for t in tensors:
        if t.device.type != "cuda" and t.device.type != "cpu":
            raise ValueError(
                f"Finufft only supports cpu and cuda tensors. Got {t.device.type}"
            )
        if t.device != device:
            raise ValueError(
                f"Some tensors are not on the same device. Got {t.device} and "
                f"{device}"
            )


def check_dtypes(data: torch.Tensor, points: torch.Tensor, name: str) -> None:
    """
    Checks that data is complex-valued
    and that points is real-valued of the same precision
    """
    complex_dtype = data.dtype
    if complex_dtype is torch.complex128:
        real_dtype = torch.float64
    elif complex_dtype is torch.complex64:
        real_dtype = torch.float32
    else:
        raise TypeError(
            f"{name} must have a dtype of torch.complex64 or torch.complex128"
        )

    if points.dtype is not real_dtype:
        raise TypeError(
            f"Points must have a dtype of {real_dtype} as {name.lower()} has a "
            f"dtype of {complex_dtype}"
        )


def check_sizes_t1(values: torch.Tensor, points: torch.Tensor) -> None:
    """
    Checks that values and points are of the same length.
    This is used in type1.
    """
    if len(values.shape) != 1:
        raise ValueError("values must be a 1d array")

    if len(points.shape) == 1:
        if len(values) != len(points):
            raise ValueError("The same number of points and values must be supplied")
    elif len(points.shape) == 2:
        if points.shape[0] not in {1, 2, 3}:
            raise ValueError(f"Points can be at most 3d, got {points.shape[0]} instead")
        if len(values) != points.shape[1]:
            raise ValueError("The same number of points and values must be supplied")
    else:
        raise ValueError("The points tensor must be 1d or 2d")


def check_output_shape(ndim: int, output_shape: Union[int, Tuple[int, ...]]) -> None:
    """
    Checks that output_shape is either an int or a tuple of ints
    """
    if isinstance(output_shape, int):
        if ndim != 1:
            raise ValueError(
                f"output_shape must be a tuple of length {ndim} for {ndim}d NUFFT"
            )
        if output_shape <= 0:
            raise ValueError("Got output_shape that was not positive integer")
    else:
        if len(output_shape) != ndim:
            raise ValueError(f"output_shape must be of length {ndim} for {ndim}d NUFFT")
        for i in output_shape:
            if i <= 0:
                raise ValueError("Got output_shape that was not positive integer")


def check_sizes_t2(targets: torch.Tensor, points: torch.Tensor) -> None:
    """
    Checks that targets and points are of the same dimension.
    This is used in type2.
    """
    targets_dim = len(targets.shape)
    if len(points.shape) == 1:
        points_dim = 1
    elif len(points.shape) == 2:
        points_dim = points.shape[0]
    else:
        raise ValueError("The points tensor must be 1d or 2d")

    if points_dim not in {1, 2, 3}:
        raise ValueError(f"Points can be at most 3d, got {points_dim} instead")

    if targets_dim != points_dim:
        raise ValueError(
            f"For type 2 {points_dim}d FINUFFT, targets must be a {points_dim}d tensor"
        )
