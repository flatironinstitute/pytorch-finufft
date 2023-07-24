"""
Implementations of the corresponding Autograd functions
"""

from typing import Optional, Tuple, Union

import finufft
import torch

########################################################################################
# Common error handling/ checking
########################################################################################


def _common_type_checks(
    points: Optional[torch.Tensor],
    values: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
) -> None:
    """
    Completes the type checks common to all three types of NUFFT

    Parameters
    ----------
    points : Optional[torch.Tensor]
        The points Tensor from the original call to the Function
    values : Optional[torch.Tensor]
        The values Tensor from the original call to the Function
    targets : Optional[torch.Tensor]
        The targets Tensor from the original call to the Function

    Raises
    ------
    TypeError
        In the case that points is not a torch.Tensor
    TypeError
        In the case that values is not a torch.Tensor
    TypeError
        In the case that targets is not a torch.Tensor
    TypeError
        In the case that points is not float-valued when it should be
    TypeError
        In the case that values is not complex-valued when it should be
    TypeError
        In the case that targets is not float-valued when it should be
    TypeError
        In the case that targets is not complex-valued when it should be
    """

    type3 = points is not None and values is not None and targets is not None

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
    """
    Performs type, size, and precision checks for type 1 FINUFFT calls.

    Parameters
    ----------
    points : torch.Tensor
        The points Tensor from the call to the Function
    values : torch.Tensor
        The values Tensor from the call to the Function

    Raises
    ------
    TypeError
        In the case that the precisions do not match between points and values (all should be single precision)
    TypeError
        In the case that the precisions do not match between points and values (all should be double precision)
    """
    _common_type_checks(points, values, None)

    # Ensure precisions are lined up
    if points.dtype is torch.float32 and values.dtype is not torch.complex64:
        raise TypeError(
            "Precisions must match; since points is torch.float32, values must be torch.complex64 for single precision"
        )

    if values.dtype is torch.complex64 and points.dtype is not torch.float32:
        raise TypeError(
            "Precisions must match; since points is torch.complex64, values must be torch.float32 for single precision"
        )

    if points.dtype is torch.float64 and values.dtype is not torch.complex128:
        raise TypeError(
            "Precisions must match; since points is torch.float64, values must be torch.complex128 for double precision"
        )

    if values.dtype is torch.complex128 and points.dtype is not torch.float64:
        raise TypeError(
            "Precisions must match; since values is torch.complex1238, points must be torch.float64 for single precision"
        )

    # Check that sizes match
    if not (len(points) == len(values)):
        raise ValueError("Both points and values must have the same length.")


def _type2_checks(points: torch.Tensor, targets: torch.Tensor) -> None:
    """
    Performs type, size, and precision checks for type 2 FINUFFT calls.

    Parameters
    ----------
    points : torch.Tensor
        The points Tensor from the call to the Function
    targets : torch.Tensor
        The targets Tensor from the call to the Function

    Raises
    ------
    TypeError
        In the case that the precisions do not match between points and targets (all should be single precision)
    TypeError
        In the case that the precisions do not match between points and targets (all should be double precision)
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
    """
    Performs type, size, and precision checks for type 3 FINUFFT calls.

    Parameters
    ----------
    points : torch.Tensor
        The points Tensor from the call to the Function
    values : torch.Tensor
        The values Tensor from the call to the Function
    targets : torch.Tensor
        The targets Tensor from the call to the Function

    Raises
    ------
    TypeError
        In the case that the precisions do not match between points, values, targets (all should be single precision)
    TypeError
        In the case that the precisions do not match between points, values, targets (all should be double precision)
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


########################################################################################
# 1d Functions
########################################################################################


class finufft1D1(torch.autograd.Function):
    """
    FINUFFT 1D problem type 1
    """

    @staticmethod
    def forward(
        ctx,
        points: torch.Tensor,
        values: torch.Tensor,
        output_shape: int,
        out: Optional[torch.Tensor] = None,
        fftshift: bool = False,
        **finufftkwargs: Optional[str],
    ) -> torch.Tensor:
        """
        Evaluates the Type 1 NUFFT on the inputs.

        NOTE: By default, the ordering is set to match that of Pytorch,
         Numpy, and Scipy's FFT APIs. To match the mode ordering
         native to FINUFFT, set `fftshift = True`.
        ```
                M-1
        f[k1] = SUM c[j] exp(+/-i k1 x(j))
                j=0

            for -N1/2 <= k1 <= (N1-1)/2
        ```

        Parameters
        ----------
        ctx : TODO [WHAT IS THE TYPE OF THIS???]
            PyTorch context object
        points : torch.Tensor
            The non-uniform points x_j. Valid only between -3pi and 3pi.
        values : torch.Tensor
            The source strengths c_j.
        output_shape : int
            Number of Fourier modes to use in the computation (which
            coincides with the length of the resultant array).
        out : Optional[torch.Tensor], optional
            Array to populate with result in-place, by default None
        fftshift : bool, optional
            If True, centers the 0 mode in the resultant torch.Tensor; by default False
        **finufftkwargs : Optional[str] TODO
            TODO  -- how to document?

        Returns
        -------
        torch.Tensor
            The resultant array f[k]

        Raises
        ------
        ValueError
            In the case that the mode ordering is double-specified with both
            fftshift and the kwarg modeord (only one should be provided).
        """
        _type1_checks(points, values)

        finufftkwargs = {k: v for k, v in finufftkwargs.items()}
        _mode_ordering = finufftkwargs.pop("modeord", 1)
        _i_sign = finufftkwargs.pop("isign", -1)

        if fftshift:
            if _mode_ordering != 1:
                raise ValueError(
                    "Double specification of ordering; only one of fftshift and modeord should be provided"
                )
            _mode_ordering = 0

        ctx.isign = _i_sign
        ctx.mode_ordering = _mode_ordering
        ctx.finufftkwargs = finufftkwargs

        ctx.save_for_backward(points, values)

        finufft_out = finufft.nufft1d1(
            points.data.numpy(),
            values.data.numpy(),
            output_shape,
            modeord=_mode_ordering,
            isign=_i_sign,
            **finufftkwargs,
        )

        return torch.from_numpy(finufft_out).type(values.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Implements gradients for backward mode automatic differentiation

        Parameters
        ----------
        ctx : TODO
            TODO PyTorch context object
        grad_output : torch.Tensor
            TODO VJP output

        Returns
        -------
        TODO [type]
            Tuple of derivatives with respect to each input
        """

        _i_sign = ctx.isign
        _mode_ordering = ctx.mode_ordering
        finufftkwargs = ctx.finufftkwargs

        points, values = ctx.saved_tensors
        grad_points = grad_values = None

        if ctx.needs_input_grad[0]:
            # w.r.t. the points x_j

            # k_ramp = torch.arange(0, grad_output.shape[-1], dtype=points.dtype)
            # if _mode_ordering == 0:
            #     # NOTE: kramp should be fft-shifted in this case
            #     k_ramp = torch.fft.fftshift(k_ramp)

            # ramped_grad_output = k_ramp * grad_output

            # np_points = (points.data).numpy()
            # np_grad_output = (ramped_grad_output.data).numpy()

            # grad_points = torch.from_numpy(
            #     finufft.nufft1d2(
            #         np_points,
            #         np_grad_output,
            #         isign=(-1 * _i_sign),
            #         modeord=_mode_ordering,
            #         **finufftkwargs,
            #     )
            # ).to(values.dtype)

            # grad_points *= values

            pass

        if ctx.needs_input_grad[1]:
            # w.r.t. the values c_j
            np_points = (points.data).numpy()
            np_grad_output = (grad_output.data).numpy()

            grad_values = torch.from_numpy(
                finufft.nufft1d2(
                    np_points,
                    np_grad_output,
                    isign=(-1 * _i_sign),
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            ).to(values.dtype)

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
        """
        Evaluates the Type 2 NUFFT on the inputs.

        NOTE: By default, the ordering is set to match that of Pytorch,
         Numpy, and Scipy's FFT APIs. To match the mode ordering
         native to FINUFFT, set `fftshift = True`.

        ```
        c[j] = SUM f[k1] exp(+/-i k1 x(j))
            k1

            for j = 0, ..., M-1, where the sum is over -N1/2 <= k1 <= (N1-1)/2
        ```

        Parameters
        ----------
        ctx : TODO [WHAT IS THE TYPE OF THIS?]
            PyTorch context object
        points : torch.Tensor
            The non-uniform points `x_j`. Valid only between -3pi and 3pi.
        targets : torch.Tensor
            The target Fourier mode coefficients `f_k`.
        out : Optional[torch.Tensor], optional
            Array to take the result in-place, by default None
        fftshift : bool, optional
            If True, centers the 0 mode in the resultant array, by default False

        Returns
        -------
        torch.Tensor
            The resultant array `c[j]`

        Raises
        ------
        ValueError
            In the case that the mode ordering is double-specified with both
            fftshift and the kwarg modeord (only one should be provided).
        """
        _type2_checks(points, targets)

        finufftkwargs = {k: v for k, v in finufftkwargs.items()}
        _mode_ordering = finufftkwargs.pop("modeord", 1)
        _i_sign = finufftkwargs.pop("isign", 1)

        if fftshift:
            if _mode_ordering != 1:
                raise ValueError(
                    "Double specification of ordering; only one of fftshift and modeord should be provided"
                )
            _mode_ordering = 0

        ctx.isign = _i_sign
        ctx.mode_ordering = _mode_ordering
        ctx.fftshift = fftshift
        ctx.finufftkwargs = finufftkwargs

        ctx.save_for_backward(points, targets)

        finufft_out = finufft.nufft1d2(
            points.detach().numpy(),
            targets.detach().numpy(),
            modeord=_mode_ordering,
            isign=_i_sign,
            **finufftkwargs,
        )

        return torch.from_numpy(finufft_out)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Implements gradients for backward mode automatic differentiation

        Parameters
        ----------
        ctx : TODO
            TODO PyTorch context object
        grad_output : torch.Tensor
            TODO VJP output

        Returns
        -------
        TODO
            Tuple of derivatives with respect to each input
        """
        _i_sign = ctx.isign
        _mode_ordering = ctx.mode_ordering
        _fftshift = ctx.fftshift
        _finufftkwargs = ctx.finufftkwargs

        points, targets = ctx.saved_tensors

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
        """
        Evaluates the Type 3 NUFFT on the inputs.

        ```
            M-1
        f[k] = SUM c[j] exp(+/-i s[k] x[j]),
            j=0

            for k = 0, ..., N-1
        ```

        Parameters
        ----------
        ctx : TODO
            PyTorch context object
        points : torch.Tensor
            The non-uniform points x_j.
        values : torch.Tensor
            The source strengths c_j.
        targets : torch.Tensor
            The non-uniform target points s_k.
        out : Optional[torch.Tensor]
            Array to populate with result in-place, by default None
        **finufftkwargs : Optional[str]
            TODO -- how to document

        Returns
        -------
        torch.Tensor
            The resultant array f[k]
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
            points.data.numpy(),
            values.data.numpy(),
            targets.data.numpy(),
            isign=_i_sign,
            **finufftkwargs,
        )

        return torch.from_numpy(finufft_out)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Implements gradients for backward mode automatic differentation

        Parameters
        ----------
        ctx : TODO
            TODO PyTorch context object
        grad_output : TODO
            TODO VJP output

        Returns
        -------
        TODO [type]
            Tuple of derivatives with respect to each input
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


###############################################################################
# 2d Functions
###############################################################################


class finufft2D1(torch.autograd.Function):
    """
    FINUFFT 2D problem type 1
    """

    @staticmethod
    def forward(
        ctx,
        points_x: torch.Tensor,
        points_y: torch.Tensor,
        values: torch.Tensor,
        output_shape: Union[int, tuple[int, int]],
        out: Optional[torch.Tensor] = None,
        fftshift: bool = False,
        **finufftkwargs: Optional[str],
    ) -> torch.Tensor:
        """
        TODO
        """

        # TODO -- this error handling could be better
        if points_x.dtype is not points_y.dtype:
            raise TypeError(
                "Point tensors in the x- and y-direction must have the same dtype."
            )
        _type1_checks(points_x, values)

        finufftkwargs = {k: v for k, v in finufftkwargs.items()}
        _mode_ordering = finufftkwargs.pop("modeord", 1)
        _i_sign = finufftkwargs.pop("isign", -1)

        if fftshift:
            # TODO -- this check should be done elsewhere? or error msg changed
            #   to note instead that there is a conflict in fftshift
            if _mode_ordering != 1:
                raise ValueError(
                    "Double specification of ordering; only one of fftshift and modeord should be provided"
                )
            _mode_ordering = 0

        return torch.ones(10)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Implements gradients for backward mode automatic differentiation

        Parameters
        ----------
        ctx : TODO
            TODO PyTorch context object
        grad_output : torch.Tensor
            TODO VJP output

        Returns
        -------
        TODO [type]
            Tuple of derivatives with respect to each input
        """
        pass


class finufft2D2(torch.autograd.Function):
    """
    FINUFFT 2D problem type 2
    """

    @staticmethod
    def forward(
        ctx,
    ) -> torch.Tensor:
        """
        TODO
        """

        return torch.ones(10)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Implements gradients for backward mode automatic differentiation

        Parameters
        ----------
        ctx : TODO
            TODO PyTorch context object
        grad_output : torch.Tensor
            TODO VJP output

        Returns
        -------
        TODO [type]
            Tuple of derivatives with respect to each input
        """

        pass


class finufft2D3(torch.autograd.Function):
    """
    FINUFFT 2D problem type 3
    """

    @staticmethod
    def forward(
        ctx,
    ) -> torch.Tensor:
        """
        TODO
        """

        return torch.ones(10)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Implements gradients for backward mode automatic differentiation

        Parameters
        ----------
        ctx : TODO
            TODO PyTorch context object
        grad_output : torch.Tensor
            TODO VJP output

        Returns
        -------
        TODO [type]
            Tuple of derivatives with respect to each input
        """

        pass
