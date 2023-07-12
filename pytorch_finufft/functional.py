"""
Implementations of the corresponding Autograd functions
"""

from typing import Optional, Tuple

import finufft
import torch

import pytorch_finufft


def _common_type_checks(
    points: Optional[torch.Tensor],
    values: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
) -> None:
    """Performs all type checks for FINUFFT forward calls.

    Args:
        points: torch.Tensor to be checked. Required for all types.
        values: torch.Tensor to be checked. Required for Type 1 and 3
        targets: torch.Tensor to be checked. Required for Type 2 and 3

    Raises:
        TypeError: in the case that not all inputs are torch.Tensor; in
            the case that the precisions do not line up properly; in
            the case that the datatypes are incorrect (eg. real-valued
            when the input should be complex-valued)
    """

    type3 = False
    if points is not None and values is not None and targets is not None:
        type3 = True

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
            "Precisions must match; since points is torch.float32, values must be torch.complex64 for single precision"
        )

    if points.dtype is torch.float64 and values.dtype is not torch.complex128:
        raise TypeError(
            "Precisions must match; since points is torch.float64, values must be torch.complex128 for double precision"
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
            "Precisions must match; since points is torch.float32, targets must be torch.complex64."
        )

    if points.dtype is torch.float64 and targets.dtype is not torch.complex128:
        raise TypeError(
            "Precisions must match; since points is torch.float64, targets must be torch.complex128."
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

    _common_type_checks(points, values, targets)

    if points.dtype is torch.float32 and (
        values.dtype is not torch.complex64
        or targets.dtype is not torch.float32
    ):
        raise TypeError(
            "Precisions must match; since points is torch.float32, values must be torch.complex64, and targets must be torch.float32"
        )

    if points.dtype is torch.float64 and (
        values.dtype is not torch.complex128
        or targets.dtype is not torch.float64
    ):
        raise TypeError(
            "Precisions must match; since points is torch.float64, values must be torch.complex128, and targets must be torch.float64"
        )

    return


class finufft1D1(torch.autograd.Function):
    """
    FINUFFT 1D problem type 1 (non-uniform points)
    """

    @staticmethod
    def forward(
        ctx,
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
            ctx: PyTorch context object
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

        Raises:
            ValueError: in the case that the mode ordering is double specified,
                implying a conflict (only one of modeord or fftshift should be provided)

        Returns:
            torch.Tensor: The resultant array
        """

        # TODO -- probably want to do away with
        if output_shape is None and out is None:
            output_shape = len(points)

        _type1_checks(points, values)

        finufftkwargs = {k: v for k, v in finufftkwargs.items()}
        _mode_ordering = finufftkwargs.pop("modeord", 1)
        _i_sign = finufftkwargs.pop("isign", -1)

        if fftshift:
            if _mode_ordering != 1:
                raise ValueError(
                    "Double specification of Fourier mode ordering;\
                         only one of fftshift and modeord should be given"
                )
            _mode_ordering = 0

        ctx.isign = _i_sign
        ctx.mode_ordering = _mode_ordering
        ctx.finufftkwargs = finufftkwargs

        ctx.save_for_backward(points, values)

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
    def backward(ctx, grad_output: torch.Tensor):
        """
        Implements gradients for backward mode automatic differentiation

        Args:
            ctx: Pytorch context object TODO
            grad_output: vjp output

        Returns:
            TODO : tuple of derivatives with respect to each input; only defined
                in the case the input is a torch.Tensor
        """

        print(type(ctx))

        _i_sign = ctx.isign
        _mode_ordering = ctx.mode_ordering
        finufftkwargs = ctx.finufftkwargs

        points, values = ctx.saved_tensors
        grad_points = grad_values = None

        if ctx.needs_input_grad[0]:
            # w.r.t. the points x_j
            grad_points = None  # finufft.nufft1d2()
        if ctx.needs_input_grad[1]:
            # w.r.t. the values c_j
            np_points = points.detach().numpy()
            np_grad_output = grad_output.numpy()

            grad_values = torch.from_numpy(
                finufft.nufft1d2(
                    np_points,
                    np_grad_output,
                    isign=_i_sign,
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            )

        return grad_points, grad_values, None, None, None, None


class finufft1D2(torch.autograd.Function):
    """
    FINUFFT 1d Problem type 2
    """

    @staticmethod
    def forward(
        ctx,
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
            ctx: PyTorch context object
            points: The non-uniform points x_j; valid only between -3pi and 3pi
            targets: Fourier mode coefficient tensor of length N1, where N1 may be even or odd.
            out: Array to take the output in-place
            fftshift: If true, centers the 0 mode in the resultant torch.Tensor
            **finufftkwargs: Keyword arguments
                # TODO -- link the one FINUFFT page regarding keywords, and note also isign and eps

        Returns:
            torch.Tensor(complex[M] or complex[ntransf, M]): The resulting array
        """
        _type2_checks(points, targets)

        finufftkwargs = {k: v for k, v in finufftkwargs.items()}
        _mode_ordering = finufftkwargs.pop("modeord", 1)
        _i_sign = finufftkwargs.pop("isign", 1)

        if fftshift:
            _mode_ordering = 0

        ctx.isign = _i_sign
        ctx.mode_ordering = _mode_ordering
        ctx.fftshift = fftshift
        ctx.finufftkwargs = finufftkwargs

        finufft_out = finufft.nufft1d2(
            points.numpy(),
            targets.numpy(),
            modeord=_mode_ordering,
            isign=_i_sign,
            **finufftkwargs,
        )

        return torch.from_numpy(finufft_out)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Implements gradients for backward mode autograd

        Args:
            ctx: Pytorch context object TODO
            grad_output: left-hand multiplicand in VJP TODO

        Returns:
            Tuple of derivatives with respect to each input to the forward
                method.
        """
        _i_sign = ctx.isign
        _mode_ordering = ctx.mode_ordering
        _fftshift = ctx.fftshift
        _finufftkwargs = ctx.finufftkwargs

        grad_points = grad_targets = None

        if ctx.needs_input_grad[0]:
            grad_points = None
        if ctx.needs_input_grad[1]:
            grad_targets = None

        return grad_points, grad_targets, None, None, None


class finufft1D3(torch.autograd.Function):
    """
    FINUFFT 1d Problem type 3
    """

    @staticmethod
    def forward(
        ctx,
        points: torch.Tensor,
        values: torch.Tensor,
        targets: torch.Tensor,
        out: Optional[torch.Tensor] = None,
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
            ctx: Pytorch context object or None
            points: The non-uniform points x_j; valid only between -3pi and 3pi
            values: The source strengths c_j
            targets: Fourier mode coefficient tensor of length N1, where N1 may be even or odd.
            out: Array to take the output in-place
            **finufftkwargs: Keyword arguments
                # TODO -- link the one FINUFFT page regarding keywords, etc

        Returns:
            torch.Tensor: The resultant array
        """
        _type3_checks(points, values, targets)

        # NB: no mode ordering kwarg for type 3
        finufftkwargs = {k: v for k, v in finufftkwargs.items()}
        _i_sign = finufftkwargs.pop("isign", -1)

        # NOTE: this is passed in as None in the test suite
        if ctx is not None:
            ctx.isign = _i_sign
            ctx.finufftkwargs = finufftkwargs

        finufft_out = finufft.nufft1d3(
            points.numpy(),
            values.numpy(),
            targets.numpy(),
            isign=_i_sign,
            **finufftkwargs,
        )

        return torch.from_numpy(finufft_out)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Implements gradients for backward mode autograd

        Args:
            ctx: Pytorch context object
            grad_output: left-hand multiplicand in VJP

        Returns:
            Tuple of derivatives with respect to each input to the forward
                method (here, 5).
        """
        _i_sign = ctx.isign
        _mode_ordering = ctx.mode_ordering
        _fftshift = ctx.fftshift
        _finufftkwargs = ctx.finufftkwargs

        grad_points = grad_targets = None

        if ctx.needs_input_grad[0]:
            grad_points = None
        if ctx.needs_input_grad[1]:
            grad_targets = None

        return grad_points, grad_targets, None, None, None
