from typing import Tuple, Union

import torch

_COORD_CHAR_TABLE = {0: "x", 1: "y", 2: "z"}


##############################################################################
# Common error handling/ checking
##############################################################################


def _type1_checks(
    points_tuple: tuple[torch.Tensor, ...],
    values: torch.Tensor,
    output_shape: Union[int, tuple[int, ...]],
) -> None:
    """
    Performs all checks for type 1

    Parameters
    ----------
    points_tuple : Tuple[torch.Tensor, ...]
        _description_
    values : torch.Tensor
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    # Ensure that values is complex
    if not torch.is_complex(values):
        raise TypeError("Got values that is not complex-valued")

    # Base the dtype and precision checks off of that of values
    complex_dtype = values.dtype
    real_dtype = (
        torch.float32 if complex_dtype is torch.complex64 else torch.float64
    )

    # Determine if 1, 2, or 3d and figure out if points, points_x, points_y, etc
    dimension = len(points_tuple)

    # Values must be 1d
    if len(values.shape) != 1:
        raise ValueError("values must be a 1d array")

    common_len = -1
    for i in range(dimension):
        coord_char = "" if dimension == 1 else ("_" + _COORD_CHAR_TABLE[i])

        if i == 0:
            common_len = len(points_tuple[i])
        else:  # Ensure all points array are of the same length
            if len(points_tuple[i]) != common_len:
                raise ValueError(
                    "Got points arrays which are not all the same length"
                )

        # Ensure all points arrays are 1d
        if len(points_tuple[i].shape) != 1:
            raise ValueError(
                "Got points" + coord_char + "that is not a 1d tensor"
            )

        # Ensure all points have the same type and correct precision
        if points_tuple[i].dtype is not real_dtype:
            raise TypeError(
                "Got points"
                + coord_char
                + " that is not "
                + str(real_dtype)
                + " valued; points"
                + coord_char
                + " must also be the same precision as values."
            )

    if len(values) != common_len:
        raise ValueError(
            "Got values that is not the same length as the points arrays."
        )

    # TODO -- is this worth keeping here or can be removed?
    if type(output_shape) is int:
        if not output_shape > 0:
            raise ValueError("Got output_shape that was not positive integer")
    else:
        for i in output_shape:
            if not i > 0:
                raise ValueError(
                    "Got output_shape that was not positive integer"
                )


def _type2_checks(
    points_tuple: tuple[torch.Tensor, ...], targets: torch.Tensor
) -> None:
    """
    _summary_

    Parameters
    ----------
    points_tuple : Tuple[torch.Tensor, ...]
        _description_
    targets : torch.Tensor
        _description_

    Raises
    ------
    TypeError
        _description_
    ValueError
        _description_
    TypeError
        _description_
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
            "For type 2 "
            + str(dimension)
            + "d FINUFFT, targets must be a "
            + str(dimension)
            + "d tensor"
        )

    coord_char = ""

    # Check dtypes (complex vs. real) on the inputs
    for i in range(dimension):
        coord_char = "" if dimension == 1 else ("_" + _COORD_CHAR_TABLE[i])

        if points_tuple[i].dtype is not real_dtype:
            raise TypeError(
                "Got points"
                + coord_char
                + " that is not "
                + str(real_dtype)
                + " valued; points"
                + coord_char
                + " must be the same precision as targets."
            )

    # Ensure that each points tensor is of the same length of the
    #  corresponding dimension of values
    s = targets.shape
    for i in range(dimension):
        if s[i] != len(points_tuple[i]):
            raise ValueError(
                "points"
                + coord_char
                + " and the "
                + str(i)
                + "th dimension must be of the same length"
            )


def _type3_checks(
    points_tuple: tuple[torch.Tensor, ...],
    values: torch.Tensor,
    targets_tuple: tuple[torch.Tensor, ...],
) -> None:
    # raise ValueError("Not yet implemented!")
    pass
