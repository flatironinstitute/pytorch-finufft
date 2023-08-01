from typing import Union

import torch

_COORD_CHAR_TABLE = "xyz"


##############################################################################
# Common error handling/ checking
##############################################################################

# TODO -- dependencies and build system in the pyproject.toml


def _type1_checks(
    points_tuple: tuple[torch.Tensor, ...],
    values: torch.Tensor,
    output_shape: Union[int, tuple[int, ...]],
) -> None:
    """
    TODO

    Parameters
    ----------
    points_tuple : tuple[torch.Tensor, ...]
        Tuples of points inputs, eg, (points,) or (points_x, points_y)
    values : torch.Tensor
        Values tensor
    output_shape : Union[int, tuple[int, ...]]
        Output shape either from the in-place array or directly passed

    Raises
    ------
    TypeError
        In the case that values is not complex-valued
    ValueError
        In the case that values is not a 1d tensor
    ValueError
        In the case that a points_{x,y,z} is not a 1d tensor
    ValueError
        In the case that a points_{x,y,z} is not of the same length as values
    TypeError
        In the case that a points_{x,y,z} is not real-valued or of the
        same precision as values
    ValueError
        In the case of a malformed output_shape
    ValueError
        In the case of a malformed output_shape
    """

    # Ensure that values is complex
    if not torch.is_complex(values):
        raise TypeError("Got values that is not complex-valued")

    # Base the dtype and precision checks off of that of values
    complex_dtype = values.dtype
    real_dtype = (
        torch.float32 if complex_dtype is torch.complex64 else torch.float64
    )

    # Determine if 1, 2, or 3d and figure out if points, points_x, points_y
    dimension = len(points_tuple)

    # Values must be 1d
    if len(values.shape) != 1:
        raise ValueError("values must be a 1d array")

    len_values = len(values)
    for i in range(dimension):
        coord_char = "" if dimension == 1 else ("_" + _COORD_CHAR_TABLE[i])

        # Ensure all points arrays are 1d
        if len(points_tuple[i].shape) != 1:
            raise ValueError(f"Got points{coord_char} that is not a 1d tensor")

        if len(points_tuple[i]) != len_values:
            raise ValueError(
                f"Got points{coord_char} of a different length than values"
            )

        # Ensure all points have the same type and correct precision
        if points_tuple[i].dtype is not real_dtype:
            raise TypeError(
                f"Got points{coord_char} that is not {real_dtype} valued; points{coord_char} must also be the same precision as values.",
            )

    if type(output_shape) is int:
        if not output_shape > 0:
            raise ValueError("Got output_shape that was not positive integer")
    else:
        # In this case, output_shape is a tuple ergo iterable
        for i in output_shape:
            if not i > 0:
                raise ValueError(
                    "Got output_shape that was not positive integer"
                )

    _device_assertions(values, points_tuple)

    return


def _type2_checks(
    points_tuple: tuple[torch.Tensor, ...], targets: torch.Tensor
) -> None:
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
    real_dtype = (
        torch.float32 if complex_dtype is torch.complex64 else torch.float64
    )

    dimension = len(points_tuple)
    targets_dim = len(targets.shape)

    if dimension != targets_dim:
        raise ValueError(
            f"For type 2 {dimension}d FINUFFT, targets must be a {dimension}d "
            "tensor"
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

    _device_assertions(targets, points_tuple)

    return


def _type3_checks(
    points_tuple: tuple[torch.Tensor, ...],
    values: torch.Tensor,
    targets_tuple: tuple[torch.Tensor, ...],
) -> None:
    # raise ValueError("Not yet implemented!")

    dimension = len(points_tuple)

    pass


def _device_assertions(
    leading: torch.Tensor, tensors: tuple[torch.Tensor, ...]
) -> None:
    """
    Asserts that all inputs are on the same device by checking against that
    of leading

    Parameters
    ----------
    leading : torch.Tensor
        The tensor against which the device of each tensor in tensors should
        be checked against
    tensors : tuple[torch.Tensor, ...]
        The remaining tensors to check

    Raises
    ------
    ValueError
        In the case that one of the tensors in tensors is not on the same
        device as the leading tensor
    """

    for t in tensors:
        if not t.device == leading.device:
            raise ValueError(
                "Ensure that all tensors passed to FINUFFT are on the same "
                "device"
            )
