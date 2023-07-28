"""
Implementations of the corresponding Autograd functions
"""

from typing import Any, Optional, Tuple, Union

import finufft
import torch

import pytorch_finufft._err as err

########################################################################################
# 1d Functions
########################################################################################


class finufft1D1(torch.autograd.Function):
    """
    FINUFFT 1D problem type 1
    """

    @staticmethod
    def forward(
        ctx: Any,
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
        ctx : Any
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
            If True centers the 0 mode in the resultant array, by default False
        **finufftkwargs : Optional[str] TODO
            TODO -- how to document?

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
        if out is not None:
            print("In-place results are not yet implemented")

        err._type1_checks((points,), values, output_shape)

        finufftkwargs = {k: v for k, v in finufftkwargs.items()}
        _mode_ordering = finufftkwargs.pop("modeord", 1)
        _i_sign = finufftkwargs.pop("isign", -1)

        if fftshift:
            if _mode_ordering != 1:
                raise ValueError(
                    "Double specification of ordering; only one of fftshift "
                    "and modeord should be provided"
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
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[Union[torch.Tensor, None], ...]:
        """
        Implements derivatives wrt. each argument in the forward method.

        Parameters
        ----------
        ctx : Any
            PyTorch context object.
        grad_output : torch.Tensor
            Backpass gradient output.

        Returns
        -------
        Tuple[Union[torch.Tensor, None], ...]
            A tuple of derivatives wrt. each argument in the forward method
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
            np_points = points.data.numpy()
            np_grad_output = grad_output.data.numpy()

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
        ctx: Any,
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
         native to FINUFFT, set fftshift=True.

        ```
        c[j] = SUM f[k1] exp(+/-i k1 x(j))
            k1

            for j = 0, ..., M-1, where the sum is over -N1/2 <= k1 <= (N1-1)/2
        ```

        Parameters
        ----------
        ctx : Any
            PyTorch context object
        points : torch.Tensor
            The non-uniform points x_j. Valid only between -3pi and 3pi.
        targets : torch.Tensor
            The target Fourier mode coefficients f_k.
        out : Optional[torch.Tensor], optional
            Array to take the result in-place, by default None
        fftshift : bool, optional
            If True centers the 0 mode in the resultant array, by default False

        Returns
        -------
        torch.Tensor
            The resultant array c[j]

        Raises
        ------
        ValueError
            In the case that the mode ordering is double-specified with both
            fftshift and the kwarg modeord (only one should be provided).
        """
        if out is not None:
            print("In-place results are not yet implemented")

        err._type2_checks((points,), targets)

        finufftkwargs = {k: v for k, v in finufftkwargs.items()}
        _mode_ordering = finufftkwargs.pop("modeord", 1)
        _i_sign = finufftkwargs.pop("isign", 1)

        if fftshift:
            if _mode_ordering != 1:
                raise ValueError(
                    "Double specification of ordering; only one of fftshift "
                    "and modeord should be provided"
                )
            _mode_ordering = 0

        ctx.isign = _i_sign
        ctx.mode_ordering = _mode_ordering
        ctx.fftshift = fftshift
        ctx.finufftkwargs = finufftkwargs

        ctx.save_for_backward(points, targets)

        finufft_out = finufft.nufft1d2(
            points.data.numpy(),
            targets.data.numpy(),
            modeord=_mode_ordering,
            isign=_i_sign,
            **finufftkwargs,
        )

        return torch.from_numpy(finufft_out)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[Union[torch.Tensor, None], ...]:
        """
        Implements derivatives wrt. each argument in the forward method

        Parameters
        ----------
        ctx : Any
            PyTorch context object
        grad_output : torch.Tensor
            Backpass gradient output

        Returns
        -------
        Tuple[Union[torch.Tensor, None], ...]
            Tuple of derivatives wrt. each argument in the forward method
        """
        _i_sign = ctx.isign
        _mode_ordering = ctx.mode_ordering
        finufftkwargs = ctx.finufftkwargs

        points, targets = ctx.saved_tensors

        grad_points = grad_targets = None

        if ctx.needs_input_grad[0]:
            grad_points = None
        if ctx.needs_input_grad[1]:
            np_points = points.data.numpy()
            np_grad_output = grad_output.data.numpy()

            grad_targets = torch.from_numpy(
                finufft.nufft1d1(
                    np_points,
                    np_grad_output,
                    len(np_points),
                    modeord=_mode_ordering,
                    isign=(-1 * _i_sign),
                    **finufftkwargs,
                )
            )

        return grad_points, grad_targets, None, None, None


class finufft1D3(torch.autograd.Function):
    """
    FINUFFT 1d Problem type 3
    """

    @staticmethod
    def forward(
        ctx: Any,
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
        ctx : Any
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

        if out is not None:
            print("In-place results are not yet implemented")

        err._type3_checks((points,), values, (targets,))

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
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, Any], ...]:
        """
        Implements gradients for backward mode automatic differentation

        Parameters
        ----------
        ctx : Any
            PyTorch context object
        grad_output : torch.Tensor
            Backpass gradient output

        Returns
        -------
        Tuple[Union[torch.Tensor, Any], ...]
            Tuple of derivatives with respect to each input
        """
        _i_sign = ctx.isign
        _mode_ordering = ctx.mode_ordering
        _fftshift = ctx.fftshift
        _finufftkwargs = ctx.finufftkwargs

        grad_points = grad_values = grad_targets = None

        if ctx.needs_input_grad[0]:
            grad_points = None
        if ctx.needs_input_grad[1]:
            grad_values = None
        if ctx.needs_input_grad[2]:
            grad_targets = None

        return grad_points, grad_values, grad_targets, None, None, None


###############################################################################
# 2d Functions
###############################################################################


class finufft2D1(torch.autograd.Function):
    """
    FINUFFT 2D problem type 1
    """

    @staticmethod
    def forward(
        ctx: Any,
        points_x: torch.Tensor,
        points_y: torch.Tensor,
        values: torch.Tensor,
        output_shape: Union[int, tuple[int, int]],
        out: Optional[torch.Tensor] = None,
        fftshift: bool = False,
        **finufftkwargs: Optional[str],
    ) -> torch.Tensor:
        """
        Evaluates the Type 1 NUFFT on the inputs.

        NOTE: By default, the ordering is set to match that of Pytorch,
         Numpy, and Scipy's FFT APIs. To match the mode ordering
         native to FINUFFT, set fftshift=true

        ```
                    M-1
        f[k1, k2] = SUM c[j] exp(+/-i (k1 x(j) + k2 y(j)))
                    j=0

            for -N1/2 <= k1 <= (N1-1)/2, -N2/2 <= k2 <= (N2-1)/2
        ```

        Parameters
        ----------
        ctx : Any
            Pytorch context object
        points_x : torch.Tensor
            The non-uniform points x_j. Valid only between -3pi and 3pi
        points_y : torch.Tensor
            The non-uniform points y_j. Valid only between -3pi and 3pi
        values : torch.Tensor
            The source strengths c_j.
        output_shape : Union[int, tuple[int, int]]
            Number of Fourier modes to use in the computation (which
            coincides with the dimensions of the resultant array). If just
            an integer is provided, rather than a 2-tuple, then the integer
            is taken to be the desired length in each dimension
        out : Optional[torch.Tensor], optional
            Array to populate with result in-place, by default None
        fftshift : bool, optional
            If True centers the 0 mode in the resultant array, by default False
        **finufftkwargs : Optional[str] TODO
            TODO -- how to document?

        Returns
        -------
        torch.Tensor
            The resultant array f[k1, k2]

        Raises
        ------
        ValueError
            In the case that the mode ordering is double-specified with both
            fftshift and the kwarg modeord (only one should be provided).
        """
        if out is not None:
            print("In-place results are not yet implemented")

        err._type1_checks((points_x, points_y), values, output_shape)

        finufftkwargs = {k: v for k, v in finufftkwargs.items()}
        _mode_ordering = finufftkwargs.pop("modeord", 1)
        _i_sign = finufftkwargs.pop("isign", -1)

        ctx.save_for_backward(points_x, points_y, values)

        if fftshift:
            # TODO -- this check should be done elsewhere? or error msg changed
            #   to note instead that there is a conflict in fftshift
            if _mode_ordering != 1:
                raise ValueError(
                    "Double specification of ordering; only one of fftshift and modeord should be provided"
                )
            _mode_ordering = 0

        ctx.isign = _i_sign
        ctx.mode_ordering = _mode_ordering
        ctx.finufftkwargs = finufftkwargs

        finufft_out = torch.from_numpy(
            finufft.nufft2d1(
                points_x.data.numpy(),
                points_y.data.numpy(),
                values.data.numpy(),
                output_shape,
                modeord=_mode_ordering,
                isign=_i_sign,
                **finufftkwargs,
            )
        )

        return finufft_out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        """
        Implements derivatives wrt. each argument in the forward method.

        Parameters
        ----------
        ctx : Any
            PyTorch context object
        grad_output : torch.Tensor
            Backpass gradient output

        Returns
        -------
        Tuple[Union[torch.Tensor, None], ...]
            Tuple of derivatives with respect to each input
        """

        _i_sign = ctx.isign
        _mode_ordering = ctx.mode_ordering
        finufftkwargs = ctx.finufftkwargs

        points_x, points_y, values = ctx.saved_tensors

        grad_points_x = grad_points_y = grad_values = None
        if ctx.needs_input_grad[0]:
            # wrt. points_x
            grad_points_x = None

        if ctx.needs_input_grad[1]:
            # wrt. points_y
            grad_points_y = None

        if ctx.needs_input_grad[2]:
            # wrt. values
            np_points_x = points_x.data.numpy()
            np_points_y = points_y.data.numpy()
            np_grad_output = grad_output.data.numpy()

            grad_values = torch.from_numpy(
                finufft.nufft2d2(
                    np_points_x,
                    np_points_y,
                    np_grad_output,
                    isign=(-1 * _i_sign),
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            )

        return grad_points_x, grad_points_y, grad_values, None, None, None, None


class finufft2D2(torch.autograd.Function):
    """
    FINUFFT 2D problem type 2
    """

    @staticmethod
    def forward(
        ctx: Any,
        points_x: torch.Tensor,
        points_y: torch.Tensor,
        targets: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        fftshift: bool = False,
        **finufftkwargs: Optional[str],
    ) -> torch.Tensor:
        """
        Evaluates the Type 2 NUFFT on the inputs.

        NOTE: By default, the ordering is set to match that of Pytorch,
         Numpy, and Scipy's FFT APIs. To match the mode ordering
         native to FINUFFT, set fftshift=True.

        ```
        c[j] = SUM f[k1, k2] exp(+/-i (k1 x(j) + k2 y(j)))
            k1, k2

            for j = 0, ..., M-1, where the sum is over -N1/2 <= k1 <= (N1-1)/2,
            -N2/2 <= k2 <= (N2-1)/2
        ```

        Parameters
        ----------
        ctx : Any
            Pytorch context objecy
        points_x : torch.Tensor
            The non-uniform points x_j
        points_y : torch.Tensor
            The non-uniform points y_j
        targets : torch.Tensor
            The target Fourier mode coefficients f[k1, k2]
        out : Optional[torch.Tensor], optional
            Array to take the result in-place, by default None
        fftshift : bool, optional
            If True centers the 0 mode in the resultant torch.Tensor, by default False
        **finufftkwargs : Optional[str]
            TODO -- write this and copy paste throughout

        Returns
        -------
        torch.Tensor
            The resultant array c[j]

        Raises
        ------
        ValueError
            In the case of conflicting specification of the wave-mode ordering.
        """

        if out is not None:
            print("In-place results are not yet implemented")

        # TODO -- extend checks to 2d
        err._type2_checks((points_x, points_y), targets)

        finufftkwargs = {k: v for k, v in finufftkwargs.items()}
        _mode_ordering = finufftkwargs.pop("modeord", 1)
        _i_sign = finufftkwargs.pop("isign", 1)

        if fftshift:
            if _mode_ordering != 1:
                raise ValueError(
                    "Double specification of ordering; only one of fftshift and modeord should be provided."
                )
            _mode_ordering = 0

        ctx.isign = _i_sign
        ctx.mode_ordering = _mode_ordering
        ctx.fftshift = fftshift
        ctx.finufftkwargs = finufftkwargs

        ctx.save_for_backward(points_x, points_y, targets)

        finufft_out = finufft.nufft2d2(
            points_x.data.numpy(),
            points_y.data.numpy(),
            targets.data.numpy(),
            modeord=_mode_ordering,
            isign=_i_sign,
            **finufftkwargs,
        )

        return torch.from_numpy(finufft_out)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[
        Union[torch.Tensor, None],
        Union[torch.Tensor, None],
        Union[torch.Tensor, None],
        None,
        None,
        None,
    ]:
        """
        Implements derivatives wrt. each argument in the forward method.

        Parameters
        ----------
        ctx : Any
            Pytorch context object
        grad_output : torch.Tensor
            Backpass gradient output.

        Returns
        -------
        tuple[ Union[torch.Tensor, None], ...]
            A tuple of derivatives wrt. each argument in the forward method
        """

        _i_sign = ctx.isign
        _mode_ordering = ctx.mode_ordering
        fftshift = ctx.fftshift
        finufftkwargs = ctx.finufftkwargs

        points_x, points_y, targets = ctx.saved_tensors

        grad_points_x = grad_points_y = grad_targets = None

        if ctx.needs_input_grad[0]:
            # wrt. points_x
            pass

        if ctx.needs_input_grad[1]:
            # wrt. points_y
            pass

        if ctx.needs_input_grad[2]:
            # wrt. targets

            np_points_x = points_x.data.numpy()
            np_points_y = points_y.data.numpy()

            np_grad_output = grad_output.data.numpy()

            grad_targets = torch.from_numpy(
                finufft.nufft2d1(
                    np_points_x,
                    np_points_y,
                    np_grad_output,
                    len(np_grad_output),
                    modeord=_mode_ordering,
                    isign=(-1 * _i_sign),
                    **finufftkwargs,
                )
            )

        return (
            grad_points_x,
            grad_points_y,
            grad_targets,
            None,
            None,
            None,
        )


class finufft2D3(torch.autograd.Function):
    """
    FINUFFT 2D problem type 3
    """

    @staticmethod
    def forward(
        ctx,
        points_x: torch.Tensor,
        points_y: torch.Tensor,
        values: torch.Tensor,
        targets_s: torch.Tensor,
        targets_t: torch.Tensor,
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


###############################################################################
# 3d Functions
###############################################################################


class finufft3D1(torch.autograd.Function):
    """
    FINUFFT 3D problem type 1
    """

    @staticmethod
    def forward(
        ctx: Any,
        points_x: torch.Tensor,
        points_y: torch.Tensor,
        points_z: torch.Tensor,
        values: torch.Tensor,
        output_shape: Union[int, tuple[int, int]],
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
        f[k1, k2, k3] = SUM c[j] exp(+/-i (k1 x(j) + k2 y(j) + k3 z(j)))
                        j=0

            for -N1/2 <= k1 <= (N1-1)/2, -N2/2 <= k2 <= (N2-1)/2, -N3/2 <= k3 <=
            (N3-1)/2 
        ```

        Parameters
        ----------
        ctx : Any
            Pytorch context object
        points_x : torch.Tensor
            The non-uniform points x_j. Valid only between -3pi and 3pi.
        points_y : torch.Tensor
            The non-uniform points y_j. Valid only between -3pi and 3pi.
        points_z : torch.Tensor
            The non-uniform points z_j. Valid only between -3pi and 3pi.
        values : torch.Tensor
            The source strengths c_j.
        output_shape : Union[int, tuple[int, int, int]]
            The number of Fourier modes to use in the computation (which 
            coincides with the length of the resultant array in each
            corresponding direction). If only an integer is provided
            rather than a tuple, it is taken as the number of modes in 
            each direction.
        out : Optional[torch.Tensor], optional
            Array to populate with result in-place, by default None
        fftshift : bool, optional
            If True centers the 0 mode in the resultant array, by default False
        **finufftkwargs : Optional[str] TODO
            TODO -- how to document?

        Returns
        -------
        torch.Tensor
            The resultant array f[k1, k2, k3]

        Raises
        ------
        ValueError
            In the case that the mode ordering is double-specified with both
            fftshift and the kwarg modeord (only one should be provided).
        """

        if out is not None:
            print("In-place results are not yet implemented")

        err._type1_checks((points_x, points_y, points_z), values, output_shape)

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

        ctx.save_for_backward(points_x, points_y, points_z, values)

        ctx.isign = _i_sign
        ctx.mode_ordering = _mode_ordering
        ctx.finufftkwargs = finufftkwargs

        finufft_out = torch.from_numpy(
            finufft.nufft3d1(
                points_x.data.numpy(),
                points_y.data.numpy(),
                points_z.data.numpy(),
                values.data.numpy(),
                output_shape,
                modeord=_mode_ordering,
                isign=_i_sign,
                **finufftkwargs,
            )
        )

        return finufft_out

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[Union[torch.Tensor, None], ...]:
        """
        Implements derivatives wrt. each argument in the forward method.

        Parameters
        ----------
        ctx : Any
            Pytorch context object.
        grad_output : torch.Tensor
            Backpass gradient output

        Returns
        -------
        tuple[Union[torch.Tensor, None], ...]
            A tuple of derivatives wrt. each argument in the forward method
        """
        _i_sign = ctx.isign
        _mode_ordering = ctx.mode_ordering
        finufftkwargs = ctx.finufftkwargs

        points_x, points_y, points_z, values = ctx.saved_tensors

        grad_points_x = grad_points_y = grad_points_z = grad_values = None

        if ctx.needs_input_grad[0]:
            grad_points_x = None

        if ctx.needs_input_grad[1]:
            grad_points_y = None

        if ctx.needs_input_grad[2]:
            grad_points_y = None

        if ctx.needs_input_grad[3]:
            np_points_x = points_x.data.numpy()
            np_points_y = points_y.data.numpy()
            np_points_z = points_z.data.numpy()
            np_grad_output = grad_output.data.numpy()

            grad_values = torch.from_numpy(
                finufft.nufft3d2(
                    np_points_x,
                    np_points_y,
                    np_points_z,
                    np_grad_output,
                    isign=(-1 * _i_sign),
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            )

        return (
            grad_points_x,
            grad_points_y,
            grad_points_z,
            grad_values,
            None,
            None,
            None,
            None,
        )


class finufft3D2(torch.autograd.Function):
    """
    FINUFFT 3D problem type 2
    """

    @staticmethod
    def forward(
        ctx: Any,
        points_x: torch.Tensor,
        points_y: torch.Tensor,
        points_z: torch.Tensor,
        targets: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        fftshift: bool = False,
        **finufftkwargs,
    ) -> torch.Tensor:
        """
        Evalutes the Type 2 NUFFT on the inputs

        NOTE: By default, the ordering is set to match that of Pytorch,
         Numpy, and Scipy's FFT APIs. To match the mode ordering
         native to FINUFFT, set fftshift=True.
        ```
        c[j] = SUM f[k1, k2, k3] exp(+/-i (k1 x(j) + k2 y(j) + k3 z(j)))
               k1, k2, k3

            for j = 0, ..., M-1, where the sum is over -N1/2 <= k1 <= (N1-1)/2,
            -N2/2 <= k2 <= (N2-1)/2, -N3/2 <= k3 <= (N3-1)/2
        ```

        Parameters
        ----------
        ctx : Any
            Pytorch context object
        points_x : torch.Tensor
            The non-uniform points x_j. Valid only between -3pi and 3pi.
        points_y : torch.Tensor
            The non-uniform points y_j. Valid only between -3pi and 3pi.
        points_z : torch.Tensor
            The non-uniform points z_j. Valid only between -3pi and 3pi.
        targets : torch.Tensor
            The target Fourier mode coefficients f_{k1, k2, k3}
        out : Optional[torch.Tensor], optional
            Array to use for in-place result, by default None
        fftshift : bool, optional
            If True centers the 0 mode in the resultant array, by default False
        **finufftkwargs : TODO
            TODO -- how to document?

        Returns
        -------
        torch.Tensor
            The resultant array c[j]

        Raises
        ------
        ValueError
            In the case that the mode ordering is double-specified with both
            fftshift and the kwarg modeord (only one should be provided).
        """
        if out is not None:
            print("In-place results are not yet implemented")

        err._type2_checks((points_x, points_y, points_z), targets)

        finufftkwargs = {k: v for k, v in finufftkwargs.items()}
        _mode_ordering = finufftkwargs.pop("modeord", 1)
        _i_sign = finufftkwargs.pop("isign", 1)

        if fftshift:
            if _mode_ordering != 1:
                raise ValueError(
                    "Double specification of ordering; only one of fftshift "
                    "and modeord should be provided."
                )
            _mode_ordering = 0

        ctx.isign = _i_sign
        ctx.mode_ordering = _mode_ordering
        ctx.fftshift = fftshift
        ctx.finufftkwargs = finufftkwargs

        ctx.save_for_backward(points_x, points_y, points_z, targets)

        finufft_out = finufft.nufft3d2(
            points_x.data.numpy(),
            points_y.data.numpy(),
            points_z.data.numpy(),
            targets.data.numpy(),
            modeord=_mode_ordering,
            isign=_i_sign,
            **finufftkwargs,
        )

        return torch.from_numpy(finufft_out)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[
        Union[torch.Tensor, None],
        Union[torch.Tensor, None],
        Union[torch.Tensor, None],
        Union[torch.Tensor, None],
        None,
        None,
        None,
    ]:
        """
        Implements derivatives wrt. each argument in the forward method

        Parameters
        ----------
        ctx : Any
            Pytorch context object
        grad_output : torch.Tensor
            Backpass gradient output

        Returns
        -------
        tuple[Union[torch.Tensor, None], ...]
            Tuple of derivatives wrt. each argument in the forward method
        """
        _i_sign = ctx.isign
        _mode_ordering = ctx.mode_ordering
        finufftkwargs = ctx.finufftkwargs

        points_x, points_y, points_z, values = ctx.saved_tensors

        grad_points_x = grad_points_y = grad_points_z = grad_values = None

        if ctx.needs_input_grad[0]:
            grad_points_x = None

        if ctx.needs_input_grad[1]:
            grad_points_y = None

        if ctx.needs_input_grad[2]:
            grad_points_y = None

        if ctx.needs_input_grad[3]:
            np_points_x = points_x.data.numpy()
            np_points_y = points_y.data.numpy()
            np_points_z = points_z.data.numpy()
            np_grad_output = grad_output.data.numpy()

            grad_values = torch.from_numpy(
                finufft.nufft3d1(
                    np_points_x,
                    np_points_y,
                    np_points_z,
                    np_grad_output,
                    len(np_grad_output),
                    isign=(-1 * _i_sign),
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            )

        return (
            grad_points_x,
            grad_points_y,
            grad_points_z,
            grad_values,
            None,
            None,
            None,
        )


class finufft3D3(torch.autograd.Function):
    """
    FINUFFT 3D problem type 3
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
    def backward(ctx: Any, grad_output: torch.Tensor):
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
