"""
Implementations of the corresponding Autograd functions
"""

from typing import Optional

import finufft
import torch


def _type1_checks(points: torch.Tensor, values: torch.Tensor) -> None:
    """Performs all type and size checks for the type 1 FINUFFT calls,
    independent of dimension.

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

    Returns:
        None
    """
    # Check that both are Tensors
    if not (
        isinstance(points, torch.Tensor) and isinstance(values, torch.Tensor)
    ):
        raise TypeError(
            "Both points and values must be torch.Tensor. Instead,\
             got points: "
            + type(points)
            + " and values: "
            + type(values)
        )

    if not torch.is_floating_point(points):
        raise TypeError(
            "points must be of floating point dtype, ie,\
                torch.float16, torch.bfloat16, torch.float32, or\
                torch.float64. Instead, got "
            + points.dtype
        )

    if not torch.is_complex(values):
        raise TypeError(
            "values must be of complex dtype, ie, torch.complex64\
                for single precision, or torch.complex128 for double\
                precision. Instead, got "
            + values.dtype
        )

    if points.dtype is torch.float32 and values.dtype is not torch.complex64:
        raise TypeError(
            "Precisions must match; since points is single precision, ie,\
            torch.float32, values must be torch.complex64."
        )

    if points.dtype is torch.float64 and values.dtype is not torch.complex128:
        raise TypeError(
            "Precisions must match; since points is single precision, ie,\
            torch.float64, values must be torch.complex128."
        )

    # Check that sizes match
    if not (len(points) == len(values)):
        raise ValueError("Both points and values must have the same length.")

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

        Returns:
            torch.Tensor(complex[N1] or complex[ntransf, N1]): The
                resultant array
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
                # TODO -- link the one FINUFFT page regarding keywords

        Raises:
            TypeError: If either or both points and targets are not torch.Tensor

        Returns:
            torch.Tensor(complex[M] or complex[ntransf, M]): The resulting array
        """
        if not (
            isinstance(points, torch.Tensor)
            and isinstance(targets, torch.Tensor)
        ):
            raise TypeError("Both points and targets must be torch.Tensor")

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
        """
        Evaluates Type 3 NUFFT on inputs

        ```
            M-1
        f[k] = SUM c[j] exp(+/-i s[k] x[j]),
            j=0

            for k = 0, ..., N-1
        ```

        Args:
            points: Nonuniform points
            values: TBD
            targets: Target
            out: In-place vector (NOT DONE)
            fftshift: Changes wave mode ordering
            **finufftkwargs: Keyword arguments to be fed as-is to FINUFFT

        Returns:
            torch.Tensor(complex[M] or complex[ntransf, M]): The resulting array
        """

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
