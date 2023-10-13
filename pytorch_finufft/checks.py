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


def check_dtypes(values: torch.Tensor, points: torch.Tensor) -> None:
    """
    Checks that values is complex-valued
    and that points is real-valued of the same precision
    """
    complex_dtype = values.dtype
    if complex_dtype is torch.complex128:
        real_dtype = torch.float64
    elif complex_dtype is torch.complex64:
        real_dtype = torch.float32
    else:
        raise TypeError(
            "Values must have a dtype of torch.complex64 or torch.complex128"
        )

    if points.dtype is not real_dtype:
        raise TypeError(
            f"Points must have a dtype of {real_dtype} as values has a dtype of "
            f"{complex_dtype}"
        )


def check_sizes(values: torch.Tensor, points: torch.Tensor) -> None:
    """
    Checks that values and points are 1d and of the same length.
    This is used in type1.
    """
    if len(values.shape) != 1:
        raise ValueError("values must be a 1d array")

    if len(points.shape) == 1:
        if len(values) != len(points):
            raise ValueError("The same number of points and values must be supplied")
    elif len(points.shape) == 2:
        if points.shape[0] not in {1, 2, 3}:
            raise ValueError(f"Points can be at most 3d, got {points.shape} instead")
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


### TODO delete the following post-consolidation

_COORD_CHAR_TABLE = "xyz"


def _type2_checks(points_tuple: torch.Tensor, targets: torch.Tensor) -> None:
    """
    Performs all type, precision, size, device, ... checks for the
    type 2 FINUFFT

    Parameters
    ----------
    points_tuple : Tuple[torch.Tensor, ...]
        A tuple of all points tensors. Eg, (points, ), or (points_x, points_y)
    targets : torch.Tensor
        The targets tensor from the forward call to FINUFFT

    Raises
    ------
    TypeError
        In the case that targets is not complex-valued
    ValueError
        In the case that targets is not of the correct shape
    TypeError
        In the case that any of the points tensors are not of the correct
        type or the correct precision
    ValueError
        In the case that the i'th dimension of targets is not of the same
        length as the i'th points tensor
    """

    if not torch.is_complex(targets):
        raise TypeError("Got values that is not complex-valued")

    complex_dtype = targets.dtype
    real_dtype = torch.float32 if complex_dtype is torch.complex64 else torch.float64

    dimension = len(points_tuple)
    targets_dim = len(targets.shape)

    if dimension != targets_dim:
        raise ValueError(
            f"For type 2 {dimension}d FINUFFT, targets must be a {dimension}d " "tensor"
        )

    coord_char = ""

    # Check dtypes (complex vs. real) on the inputs
    for i in range(dimension):
        coord_char = "" if dimension == 1 else ("_" + _COORD_CHAR_TABLE[i])

        if points_tuple[i].dtype is not real_dtype:
            raise TypeError(
                f"Got points{coord_char} that is not {real_dtype}-valued; "
                f"points{coord_char} must also be the same precision as "
                "targets."
            )
