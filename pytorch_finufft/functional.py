"""
Implementations of the corresponding Autograd functions
"""

from typing import Optional

import finufft
import torch


def _common_type_checks(
    points: Optional[torch.Tensor],
    values: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
    type3: bool = False,
) -> None:
    """Performs all type checks for FINUFFT forward calls."""
    # Check all tensors
    if points is not None and not isinstance(points, torch.Tensor):
        raise TypeError("points must be a torch.Tensor")
    elif values is not None and not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    elif targets is not None and not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a torch.Tensor")

    # Check for the correct dtype
    if points is not None and not torch.is_floating_point(points):
        raise TypeError("points should be float-valued.")
    if values is not None and not torch.is_complex(values):
        raise TypeError("values should be complex-valued.")

    if targets is not None:
        if type3 and not torch.is_floating_point(targets):
            raise TypeError("targets should be float-valued.")
        elif not type3 and not torch.is_complex(targets):
            raise TypeError("targets should be complex-valued.")


def _type1_checks(points: torch.Tensor, values: torch.Tensor) -> None:
    """Performs type, size, and precision checks for type 1 FINUFFT calls

    Args:
        points: torch.Tensor to check as satisfying requirements for
            first positional to the type 1 FINUFFT.
        values: torch.Tensor to check as satisfying requirements for
            second positional to the type 1 FINUFFT.

    Raises:
        TypeError: If one or both are not torch.Tensor; if points
            is not a floating point torch.Tensor or if values is
            not complex-valued; if there is a single/ double precision
            mismatch between points and values.
        ValueError: If the lengths of points and values are mismatched.
    """
    _common_type_checks(points, values, None)

    # Ensure precisions are lined up
    if points.dtype is torch.float32 and values.dtype is not torch.complex64:
        raise TypeError(
            "Precisions must match; since points is torch.float32, values must\
            be torch.complex64 for single precision"
        )

    if points.dtype is torch.float64 and values.dtype is not torch.complex128:
        raise TypeError(
            "Precisions must match; since points is torch.float64, values must\
            be torch.complex64 for double precision"
        )

    # Check that sizes match
    if not (len(points) == len(values)):
        raise ValueError("Both points and values must have the same length.")


def _type2_checks(points: torch.Tensor, targets: torch.Tensor) -> None:
    """Performs type, size, and precision checks for type 2 FINUFFT calls.

    Args:
        points: torch.Tensor to check as satisfying the requirements as the
            first positional to the type 2 FINUFFT
        targets: torch.Tensor to check as satisfying the requirements as the
            second positional to the type 2 FINUFFT

    Raises:
        TypeError: If there is a precision mismatch.
    """
    # Type checks
    _common_type_checks(points, None, targets)

    # Check precisions
    if points.dtype is torch.float32 and targets.dtype is not torch.complex64:
        raise TypeError(
            "Precisions must match; since points is single precision, ie,\
            torch.float32, targets must be torch.complex64."
        )

    if points.dtype is torch.float64 and targets.dtype is not torch.complex128:
        raise TypeError(
            "Precisions must match; since points is double precision, ie,\
            torch.float64, targets must be torch.complex128."
        )

    return


def _type3_checks(
    points: torch.Tensor, values: torch.Tensor, targets: torch.Tensor
) -> None:
    """Performs type, size, and precision checks for type 3 FINUFFT calls.

    Args:
        points: TODO
        values: TODO
        targets: TODO

    Raises:
        TypeError: TODO
    """

    _common_type_checks(points, values, targets, True)

    if points.dtype is torch.float32 and (
        values.dtype is not torch.complex64
        or targets.dtype is not torch.complex64
    ):
        raise TypeError(
            "Precisions must match; since points is torch.float32, values\
                and targets must be torch.complex64"
        )

    if points.dtype is torch.float64 and (
        values.dtype is not torch.complex128
        or targets.dtype is not torch.float64
    ):
        raise TypeError(
            "Precisions must match; since points is torch.float64, values\
                must be torch.complex128, and targets must be torch.float64"
        )

    return


class finufft1D1(torch.autograd.Function):
    """
    FINUFFT 1D problem type 1 (non-uniform points)
    """

    @staticmethod
    def forward(
        points: torch.Tensor,
        values: torch.Tensor,
        output_shape: Optional[int] = None,
        out: Optional[torch.Tensor] = None,
        fftshift: bool = False,
        **finufftkwargs: Optional[str],
    ) -> torch.Tensor:
        """Evaluates the Type 1 NUFFT on the inputs.

        NOTE: By default, the ordering is set to match that of Pytorch,
         Numpy, and Scipy's FFT implementations. To match the mode ordering
         native to FINUFFT, set fftshift = True.
        ```
                M-1
        f[k1] = SUM c[j] exp(+/-i k1 x(j))
                j=0

            for -N1/2 <= k1 <= (N1-1)/2
        ```

        Args:
            points: The non-uniform points x_j; valid only
                between -3pi and 3pi.
            values: The source strengths c_j
            output_shape: Number of Fourier modes to use in the computation;
                should be specified if out is not given. If neither are
                given, the length of values and points will be used to
                infer output_shape.
            out: Vector to fill in-place with resulting values; should be
                provided if output_shape is not given
            fftshift: If true, centers the 0 mode in the
                resultant torch.Tensor.
            **finufftkwargs: Keyword arguments to be passed directly
                into FINUFFT Python API
                #TODO -- link the one page, and note also isign.

        Returns:
            torch.Tensor: The resultant array
        """

        if output_shape is None and out is None:
            output_shape = len(points)

        _type1_checks(points, values)

        _mode_ordering = finufftkwargs.pop("modeord", 1)
        _i_sign = finufftkwargs.pop("isign", -1)

        if fftshift:
            _mode_ordering = 0

        finufft_out = finufft.nufft1d1(
            points.numpy(),
            values.numpy(),
            output_shape,
            modeord=_mode_ordering,
            isign=_i_sign,
            **finufftkwargs,
        )

        return torch.from_numpy(finufft_out)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        raise ValueError("TBD")

    @staticmethod
    def backward(ctx, f_k: torch.Tensor):
        """
        Implements gradients for backward mode automatic differentiation
        """

        raise ValueError("TBD")


class finufft1D2(torch.autograd.Function):
    """
    FINUFFT 1d Problem type 2
    """

    @staticmethod
    def forward(
        points: torch.Tensor,
        targets: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        fftshift: bool = False,
        **finufftkwargs,
    ) -> torch.Tensor:
        """Evaluates Type 2 NUFFT on inputs

        ```
        c[j] = SUM f[k1] exp(+/-i k1 x(j))
               k1

            for j = 0, ..., M-1, where the sum is over -N1/2 <= k1 <= (N1-1)/2
        ```

        Args:
            points: The non-uniform points x_j; valid only between -3pi and 3pi
            targets: Fourier mode coefficient tensor of length N1, where N1 may be even or odd.
            out: Array to take the output in-place
            fftshift: If true, centers the 0 mode in the resultant torch.Tensor
            **finufftkwargs: Keyword arguments
                # TODO -- link the one FINUFFT page regarding keywords, and note also isign

        Returns:
            torch.Tensor(complex[M] or complex[ntransf, M]): The resulting array
        """
        _type2_checks(points, targets)

        finufft_out = finufft.nufft1d2(
            points.numpy(), targets.numpy(), modeord=1, isign=1
        )

        return torch.from_numpy(finufft_out)

    @staticmethod
    def setup_context(_):
        raise ValueError("TBD")

    @staticmethod
    def backward(_):
        raise ValueError("TBD")


class finufft1D3(torch.autograd.Function):
    """
    FINUFFT 1d Problem type 3
    """

    @staticmethod
    def forward(
        points: torch.Tensor,
        values: torch.Tensor,
        targets: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        fftshift: bool = False,
        **finufftkwargs,
    ) -> torch.Tensor:
        """Evaluates Type 3 NUFFT on inputs

        ```
            M-1
        f[k] = SUM c[j] exp(+/-i s[k] x[j]),
            j=0

            for k = 0, ..., N-1
        ```

        Args:
            points: TODO
            values: TODO
            targets: TODO
            out: TODO
            fftshift: TODO
            **finufftkwargs: TODO

        Returns:
            torch.Tensor: The resultant array
        """

        _type3_checks(points, values, targets)

        finufft_out = finufft.nufft1d3(
            points.numpy(), values.numpy(), targets.numpy(), isign=-1
        )

        return torch.from_numpy(finufft_out)

    @staticmethod
    def setup_context(_):
        raise ValueError("TBD")

    @staticmethod
    def backward(_):
        raise ValueError("TBD")
