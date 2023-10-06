"""
Implementations of the corresponding Autograd functions
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

try:
    import finufft

    FINUFFT_AVAIL = True
except ImportError:
    FINUFFT_AVAIL = False

try:
    import cufinufft

    CUFINUFFT_AVAIL = True
except ImportError:
    CUFINUFFT_AVAIL = False

if not (FINUFFT_AVAIL or CUFINUFFT_AVAIL):
    raise ImportError(
        "No FINUFFT implementation available. "
        "Install either finufft or cufinufft and ensure they are importable."
    )

import pytorch_finufft._err as err

###############################################################################
# 1d Functions
###############################################################################


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
        finufftkwargs: Dict[str, Union[int, float]] = None,
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
        fftshift : bool
            If True centers the 0 mode in the resultant array, by default False
        finufftkwargs : Dict[str, Union[int, float]]
            Additional arguments will be passed into FINUFFT. See
            https://finufft.readthedocs.io/en/latest/python.html. By default
            an empty dictionary

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

        if finufftkwargs is None:
            finufftkwargs = dict()

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
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """
        Implements derivatives wrt. each argument in the forward method.

        Parameters
        ----------
        ctx : Any
            PyTorch context object.
        grad_output : torch.Tensor
            Backpass gradient wrt. output of NUFFT operation

        Returns
        -------
        Tuple[Union[torch.Tensor, None], ...]
            A tuple of derivatives wrt. each argument in the forward method
        """
        _i_sign = -1 * ctx.isign
        _mode_ordering = ctx.mode_ordering
        finufftkwargs = ctx.finufftkwargs

        points, values = ctx.saved_tensors
        grad_points = grad_values = None

        if ctx.needs_input_grad[0]:
            # w.r.t. the points x_j

            k_ramp = torch.arange(0, grad_output.shape[-1], dtype=points.dtype) - (
                grad_output.shape[-1] // 2
            )
            if _mode_ordering != 0:
                k_ramp = torch.fft.ifftshift(k_ramp)

            # TODO analytically work out if we can simplify this *1j,
            # the below conj, and below *values
            ramped_grad_output = k_ramp * grad_output * 1j * _i_sign

            np_points = (points.data).numpy()
            np_grad_output = (ramped_grad_output.data).numpy()

            grad_points = torch.from_numpy(
                finufft.nufft1d2(
                    np_points,
                    np_grad_output,
                    isign=_i_sign,
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            ).to(values.dtype)

            grad_points = grad_points.conj()
            grad_points *= values

            grad_points = grad_points.real

        if ctx.needs_input_grad[1]:
            # w.r.t. the values c_j
            np_points = points.data.numpy()
            np_grad_output = grad_output.data.numpy()

            grad_values = torch.from_numpy(
                finufft.nufft1d2(
                    np_points,
                    np_grad_output,
                    isign=_i_sign,
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
        finufftkwargs: Dict[str, Union[int, float]] = {},
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
        fftshift : bool
            If True centers the 0 mode in the resultant array, by default False
        finufftkwargs : Dict[str, Union[int, float]]
            Additional arguments will be passed into FINUFFT. See
            https://finufft.readthedocs.io/en/latest/python.html. By default
            an empty dictionary

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
    ) -> Tuple[Union[torch.Tensor, None], ...]:
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
            # w.r.t. the points x_j

            k_ramp = torch.arange(0, targets.shape[-1], dtype=points.dtype) - (
                targets.shape[-1] // 2
            )
            if _mode_ordering != 0:
                k_ramp = torch.fft.ifftshift(k_ramp)

            # TODO analytically work out if we can simplify this *1j,
            # the below conj, and below *values
            ramped_targets = k_ramp * targets * 1j * _i_sign

            np_points = (points.data).numpy()
            np_ramped_targets = (ramped_targets.data).numpy()

            grad_points = torch.from_numpy(
                finufft.nufft1d2(
                    np_points,
                    np_ramped_targets,
                    isign=_i_sign,
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            ).to(targets.dtype)

            grad_points = grad_points.conj()
            grad_points *= grad_output

            grad_points = grad_points.real

        if ctx.needs_input_grad[1]:
            np_points = points.data.numpy()
            np_grad_output = grad_output.data.numpy()

            grad_targets = torch.from_numpy(
                finufft.nufft1d1(
                    np_points,
                    np_grad_output,
                    len(targets),
                    modeord=_mode_ordering,
                    isign=(-1 * _i_sign),
                    **finufftkwargs,
                )
            )

        return grad_points, grad_targets, None, None, None


class _finufft1D3(torch.autograd.Function):
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
        finufftkwargs: Dict[str, Union[int, float]] = {},
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
        finufftkwargs : Dict[str, Union[int, float]]
            Additional arguments will be passed into FINUFFT. See
            https://finufft.readthedocs.io/en/latest/python.html. By default
            an empty dictionary

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
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, Any], ...]:
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
        output_shape: Union[int, Tuple[int, int]],
        out: Optional[torch.Tensor] = None,
        fftshift: bool = False,
        finufftkwargs: Dict[str, Union[int, float]] = {},
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
        output_shape : Union[int, Tuple[int, int]]
            Number of Fourier modes to use in the computation (which
            coincides with the dimensions of the resultant array). If just
            an integer is provided, rather than a 2-tuple, then the integer
            is taken to be the desired length in each dimension
        out : Optional[torch.Tensor], optional
            Array to populate with result in-place, by default None
        fftshift : bool
            If True centers the 0 mode in the resultant array, by default False
        finufftkwargs : Dict[str, Union[int, float]]
            Additional arguments will be passed into FINUFFT. See
            https://finufft.readthedocs.io/en/latest/python.html. By default
            an empty dictionary

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
                    "Double specification of ordering; only one of fftshift and "
                    "modeord should be provided"
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
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
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

        _i_sign = -1 * ctx.isign
        _mode_ordering = ctx.mode_ordering
        finufftkwargs = ctx.finufftkwargs

        points_x, points_y, values = ctx.saved_tensors

        x_ramp = torch.arange(0, grad_output.shape[0], dtype=points_x.dtype) - (
            grad_output.shape[0] // 2
        )
        y_ramp = torch.arange(0, grad_output.shape[1], dtype=points_y.dtype) - (
            grad_output.shape[1] // 2
        )
        XX, YY = torch.meshgrid(x_ramp, y_ramp)

        grad_points_x = grad_points_y = grad_values = None
        if ctx.needs_input_grad[0]:
            # wrt. points_x

            if _mode_ordering != 0:
                XX = torch.fft.ifftshift(XX)

            # TODO analytically work out if we can simplify this *1j,
            # the below conj, and below *values
            ramped_grad_output = XX * grad_output * 1j * _i_sign

            np_points_x = points_x.data.numpy()
            np_points_y = points_y.data.numpy()
            np_grad_output = ramped_grad_output.data.numpy()

            grad_points = torch.from_numpy(
                finufft.nufft2d2(
                    np_points_x,
                    np_points_y,
                    np_grad_output,
                    isign=_i_sign,
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            ).to(values.dtype)

            grad_points_x = (grad_points.conj() * values).real

        if ctx.needs_input_grad[1]:
            # wrt. points_y

            if _mode_ordering != 0:
                YY = torch.fft.ifftshift(YY)

            # TODO analytically work out if we can simplify this *1j,
            # the below conj, and below *values
            ramped_grad_output = YY * grad_output * 1j * _i_sign

            np_points_x = points_x.data.numpy()
            np_points_y = points_y.data.numpy()
            np_grad_output = ramped_grad_output.data.numpy()

            grad_points = torch.from_numpy(
                finufft.nufft2d2(
                    np_points_x,
                    np_points_y,
                    np_grad_output,
                    isign=_i_sign,
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            ).to(values.dtype)

            grad_points_y = (grad_points.conj() * values).real

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
                    isign=_i_sign,
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
        finufftkwargs: Dict[str, Union[int, float]] = {},
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
        fftshift : bool
            If True centers the 0 mode in the resultant torch.Tensor, by default False
        finufftkwargs : Dict[str, Union[int, float]]
            Additional arguments will be passed into FINUFFT. See
            https://finufft.readthedocs.io/en/latest/python.html. By default
            an empty dictionary

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
                    "Double specification of ordering; only one of fftshift and "
                    "modeord should be provided."
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
    ) -> Tuple[
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
        Tuple[ Union[torch.Tensor, None], ...]
            A tuple of derivatives wrt. each argument in the forward method
        """
        _i_sign = ctx.isign
        _mode_ordering = ctx.mode_ordering
        finufftkwargs = ctx.finufftkwargs

        points_x, points_y, targets = ctx.saved_tensors

        x_ramp = torch.arange(0, targets.shape[0], dtype=points_x.dtype) - (
            targets.shape[0] // 2
        )
        y_ramp = torch.arange(0, targets.shape[1], dtype=points_y.dtype) - (
            targets.shape[1] // 2
        )
        XX, YY = torch.meshgrid(x_ramp, y_ramp)

        grad_points_x = grad_points_y = grad_targets = None

        if ctx.needs_input_grad[0]:
            # wrt. points_x
            if _mode_ordering != 0:
                XX = torch.fft.ifftshift(XX)

            # TODO analytically work out if we can simplify this *1j,
            # the below conj, and below *values
            ramped_targets = XX * targets * 1j * _i_sign

            np_points_x = points_x.data.numpy()
            np_points_y = points_y.data.numpy()
            np_ramped_targets = ramped_targets.data.numpy()

            grad_points_x = torch.from_numpy(
                finufft.nufft2d2(
                    np_points_x,
                    np_points_y,
                    np_ramped_targets,
                    isign=_i_sign,
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            ).to(targets.dtype)

            grad_points_x = grad_points_x.conj()
            grad_points_x *= grad_output

            grad_points_x = grad_points_x.real

        if ctx.needs_input_grad[1]:
            # wrt. points_y

            if _mode_ordering != 0:
                YY = torch.fft.ifftshift(YY)

            # TODO analytically work out if we can simplify this *1j,
            # the below conj, and below *values
            ramped_targets = YY * targets * 1j * _i_sign

            np_points_x = points_x.data.numpy()
            np_points_y = points_y.data.numpy()
            np_ramped_targets = ramped_targets.data.numpy()

            grad_points_y = torch.from_numpy(
                finufft.nufft2d2(
                    np_points_x,
                    np_points_y,
                    np_ramped_targets,
                    isign=_i_sign,
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            ).to(targets.dtype)

            grad_points_y = grad_points_y.conj()
            grad_points_y *= grad_output

            grad_points_y = grad_points_y.real

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
                    len(targets),
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


class _finufft2D3(torch.autograd.Function):
    """
    FINUFFT 2D problem type 3
    """

    @staticmethod
    def forward(
        ctx: Any,
        points_x: torch.Tensor,
        points_y: torch.Tensor,
        values: torch.Tensor,
        targets_s: torch.Tensor,
        targets_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluates the Type 3 NUFFT on the inputs

        Parameters
        ----------
        ctx : Any
            _description_
        points_x : torch.Tensor
            _description_
        points_y : torch.Tensor
            _description_
        values : torch.Tensor
            _description_
        targets_s : torch.Tensor
            _description_
        targets_t : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            _description_

        Raises
        ------
        ValueError
            _description_
        """

        if True:
            raise ValueError("2D3 is not implemented yet")

        return torch.ones(10)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """
        Implements gradients for backward mode automatic differentiation

        Parameters
        ----------
        ctx : Any
            TODO PyTorch context object
        grad_output : torch.Tensor
            TODO VJP output

        Returns
        -------
        Tuple[Union[torch.Tensor, None], ...]
            Tuple of derivatives with respect to each input
        """

        return None, None, None, None, None


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
        output_shape: Union[int, Tuple[int, int]],
        out: Optional[torch.Tensor] = None,
        fftshift: bool = False,
        finufftkwargs: Dict[str, Union[int, float]] = {},
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
        output_shape : Union[int, Tuple[int, int, int]]
            The number of Fourier modes to use in the computation (which
            coincides with the length of the resultant array in each
            corresponding direction). If only an integer is provided
            rather than a tuple, it is taken as the number of modes in
            each direction.
        out : Optional[torch.Tensor], optional
            Array to populate with result in-place, by default None
        fftshift : bool
            If True centers the 0 mode in the resultant array, by default False
        finufftkwargs : Dict[str, Union[int, float]]
            Additional arguments will be passed into FINUFFT. See
            https://finufft.readthedocs.io/en/latest/python.html. By default
            an empty dictionary

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
                    "Double specification of ordering; only one of fftshift and "
                    "modeord should be provided"
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
    ) -> Tuple[Union[torch.Tensor, None], ...]:
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
        Tuple[Union[torch.Tensor, None], ...]
            A tuple of derivatives wrt. each argument in the forward method
        """
        _i_sign = -1 * ctx.isign
        _mode_ordering = ctx.mode_ordering
        finufftkwargs = ctx.finufftkwargs

        points_x, points_y, points_z, values = ctx.saved_tensors

        x_ramp = torch.arange(0, grad_output.shape[0], dtype=points_x.dtype) - (
            grad_output.shape[0] // 2
        )
        y_ramp = torch.arange(0, grad_output.shape[1], dtype=points_y.dtype) - (
            grad_output.shape[1] // 2
        )
        z_ramp = torch.arange(0, grad_output.shape[2], dtype=points_z.dtype) - (
            grad_output.shape[2] // 2
        )
        XX, YY, ZZ = torch.meshgrid(x_ramp, y_ramp, z_ramp)

        grad_points_x = grad_points_y = grad_points_z = grad_values = None

        if ctx.needs_input_grad[0]:
            # wrt. points_x

            if _mode_ordering != 0:
                XX = torch.fft.ifftshift(XX)

            # TODO analytically work out if we can simplify this *1j,
            # the below conj, and below *values
            ramped_grad_output = XX * grad_output * 1j * _i_sign

            np_points_x = points_x.data.numpy()
            np_points_y = points_y.data.numpy()
            np_points_z = points_z.data.numpy()
            np_ramped_output = ramped_grad_output.data.numpy()

            grad_points_x = torch.from_numpy(
                finufft.nufft3d2(
                    np_points_x,
                    np_points_y,
                    np_points_z,
                    np_ramped_output,
                    isign=_i_sign,
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            ).to(values.dtype)

            grad_points_x = (grad_points_x.conj() * values).real

        if ctx.needs_input_grad[1]:
            if _mode_ordering != 0:
                YY = torch.fft.ifftshift(YY)

            # TODO analytically work out if we can simplify this *1j,
            # the below conj, and below *values
            ramped_grad_output = YY * grad_output * 1j * _i_sign

            np_points_x = points_x.data.numpy()
            np_points_y = points_y.data.numpy()
            np_points_z = points_z.data.numpy()
            np_ramped_output = ramped_grad_output.data.numpy()

            grad_points = torch.from_numpy(
                finufft.nufft3d2(
                    np_points_x,
                    np_points_y,
                    np_points_z,
                    np_ramped_output,
                    isign=_i_sign,
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            ).to(values.dtype)

            grad_points_y = (grad_points.conj() * values).real

        if ctx.needs_input_grad[2]:
            if _mode_ordering != 0:
                ZZ = torch.fft.ifftshift(ZZ)

            # TODO analytically work out if we can simplify this *1j,
            # the below conj, and below *values
            ramped_grad_output = ZZ * grad_output * 1j * _i_sign

            np_points_x = points_x.data.numpy()
            np_points_y = points_y.data.numpy()
            np_points_z = points_z.data.numpy()
            np_ramped_output = ramped_grad_output.data.numpy()

            grad_points_z = torch.from_numpy(
                finufft.nufft3d2(
                    np_points_x,
                    np_points_y,
                    np_points_z,
                    np_ramped_output,
                    isign=_i_sign,
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            ).to(values.dtype)

            grad_points_z = (grad_points_z.conj() * values).real

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
                    isign=_i_sign,
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
        finufftkwargs: Dict[str, Union[int, float]] = {},
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
        fftshift : bool
            If True centers the 0 mode in the resultant array, by default False
        finufftkwargs : Dict[str, Union[int, float]]
            Additional arguments will be passed into FINUFFT. See
            https://finufft.readthedocs.io/en/latest/python.html. By default
            an empty dictionary

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
    ) -> Tuple[Union[torch.Tensor, None], ...]:
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
        Tuple[Union[torch.Tensor, None], ...]
            Tuple of derivatives wrt. each argument in the forward method
        """
        _i_sign = ctx.isign
        _mode_ordering = ctx.mode_ordering
        finufftkwargs = ctx.finufftkwargs

        points_x, points_y, points_z, targets = ctx.saved_tensors

        x_ramp = torch.arange(0, targets.shape[0], dtype=points_x.dtype) - (
            targets.shape[0] // 2
        )
        y_ramp = torch.arange(0, targets.shape[0], dtype=points_x.dtype) - (
            targets.shape[0] // 2
        )
        z_ramp = torch.arange(0, targets.shape[0], dtype=points_x.dtype) - (
            targets.shape[0] // 2
        )
        XX, YY, ZZ = torch.meshgrid(x_ramp, y_ramp, z_ramp)

        grad_points_x = grad_points_y = grad_points_z = grad_values = None

        if ctx.needs_input_grad[0]:
            # wrt. points_x
            if _mode_ordering != 0:
                XX = torch.fft.ifftshift(XX)

            # TODO analytically work out if we can simplify this *1j,
            # the below conj, and below *values
            ramped_targets = XX * targets * 1j * _i_sign

            np_points_x = points_x.data.numpy()
            np_points_y = points_y.data.numpy()
            np_points_z = points_z.data.numpy()
            np_ramped_targets = ramped_targets.data.numpy()

            grad_points_x = torch.from_numpy(
                finufft.nufft3d2(
                    np_points_x,
                    np_points_y,
                    np_points_z,
                    np_ramped_targets,
                    isign=_i_sign,
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            ).to(targets.dtype)

            grad_points_x = (grad_points_x.conj() * grad_output).real

        if ctx.needs_input_grad[1]:
            # wrt. points_y
            if _mode_ordering != 0:
                YY = torch.fft.ifftshift(YY)

            # TODO analytically work out if we can simplify this *1j,
            # the below conj, and below *values
            ramped_targets = YY * targets * 1j * _i_sign

            np_points_x = points_x.data.numpy()
            np_points_y = points_y.data.numpy()
            np_points_z = points_z.data.numpy()
            np_ramped_targets = ramped_targets.data.numpy()

            grad_points_y = torch.from_numpy(
                finufft.nufft3d2(
                    np_points_x,
                    np_points_y,
                    np_points_z,
                    np_ramped_targets,
                    isign=_i_sign,
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            ).to(targets.dtype)

            grad_points_y = (grad_points_y.conj() * grad_output).real

        if ctx.needs_input_grad[2]:
            # wrt. points_z
            if _mode_ordering != 0:
                ZZ = torch.fft.ifftshift(ZZ)

            # TODO analytically work out if we can simplify this *1j,
            # the below conj, and below *values
            ramped_targets = ZZ * targets * 1j * _i_sign

            np_points_x = points_x.data.numpy()
            np_points_y = points_y.data.numpy()
            np_points_z = points_z.data.numpy()
            np_ramped_targets = ramped_targets.data.numpy()

            grad_points_z = torch.from_numpy(
                finufft.nufft3d2(
                    np_points_x,
                    np_points_y,
                    np_points_z,
                    np_ramped_targets,
                    isign=_i_sign,
                    modeord=_mode_ordering,
                    **finufftkwargs,
                )
            ).to(targets.dtype)

            grad_points_z = (grad_points_z.conj() * grad_output).real

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


class _finufft3D3(torch.autograd.Function):
    """
    FINUFFT 3D problem type 3
    """

    @staticmethod
    def forward(
        ctx: Any,
        points_x: torch.Tensor,
        points_y: torch.Tensor,
        points_z: torch.Tensor,
        values: torch.Tensor,
        targets_s: torch.Tensor,
        targets_t: torch.Tensor,
        targets_u: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        **finufftkwargs: Union[int, float],
    ) -> torch.Tensor:
        """
        TODO Description here!

        Parameters
        ----------
        ctx : Any
            PyTorch context object
        points_x : torch.Tensor
            The nonuniform source points x_j
        points_y : torch.Tensor
            The nonuniform source points y_j
        points_z : torch.Tensor
            The nonuniform source points z_j
        values : torch.Tensor
            The source strengths c_j
        targets_s : torch.Tensor
            The target Fourier mode coefficients s_k
        targets_t : torch.Tensor
            The target Fourier mode coefficients t_k
        targets_u : torch.Tensor
            The target Fourier mode coefficients u_k
        out : Optional[torch.Tensor], optional
            Array to take the result in-place, by default None
        **finufftkwargs : Union[int, float]
            Additional arguments will be passed into FINUFFT. See
            https://finufft.readthedocs.io/en/latest/python.html

        Returns
        -------
        torch.Tensor
            The resultant array f[k]

        Raises
        ------
        ValueError
            _description_
        """

        if True:
            raise ValueError("3D3 is not implemented yet")

        return torch.ones(10)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """
        Implements gradients for backward mode automatic differentiation

        Parameters
        ----------
        ctx : Any
            TODO PyTorch context object
        grad_output : torch.Tensor
            TODO VJP output

        Returns
        -------
        Tuple[Union[torch.Tensor, None], ...]
            Tuple of derivatives with respect to each input
        """

        grad_points_x = grad_points_y = grad_points_z = None

        grad_values = None

        grad_targets_s = grad_targets_t = grad_targets_u = None

        return (
            grad_points_x,
            grad_points_y,
            grad_points_z,
            grad_values,
            grad_targets_s,
            grad_targets_t,
            grad_targets_u,
            None,
            None,
        )


###############################################################################
# Consolidated forward function for all 1D, 2D, and 3D problems for nufft type 1
###############################################################################


def get_nufft_func(dim, nufft_type, device_type):
    if device_type == "cuda":
        return getattr(cufinufft, f"nufft{dim}d{nufft_type}")

    # CPU needs extra work to go to/from torch and numpy
    finufft_func = getattr(finufft, f"nufft{dim}d{nufft_type}")

    def f(*args, **kwargs):
        new_args = [arg for arg in args]
        for i in range(len(new_args)):
            if isinstance(new_args[i], torch.Tensor):
                new_args[i] = new_args[i].data.numpy()

        return torch.from_numpy(finufft_func(*new_args, **kwargs))

    return f


class finufft_type1(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        points: torch.Tensor,
        values: torch.Tensor,
        output_shape: Union[int, Tuple[int, int], Tuple[int, int, int]],
        out: Optional[torch.Tensor] = None,
        fftshift: bool = False,
        finufftkwargs: dict[str, Union[int, float]] = None,
    ):
        """
        Evaluates the Type 1 NUFFT on the inputs.

        """

        if out is not None:
            print("In-place results are not yet implemented")
            # All this requires is a check on the out array to make sure it is the
            # correct shape.

        # TODO:
        # revisit these error checks to take into account the shape of points
        # instead of passing them separately
        # make sure these checks check for consistency between output shape and
        # len(points)
        # Also need device checks
        err._type1_checks(points, values, output_shape)

        if finufftkwargs is None:
            finufftkwargs = dict()
        finufftkwargs = {k: v for k, v in finufftkwargs.items()}
        _mode_ordering = finufftkwargs.pop("modeord", 1)
        _i_sign = finufftkwargs.pop("isign", -1)

        if fftshift:
            # TODO -- this check should be done elsewhere? or error msg changed
            #   to note instead that there is a conflict in fftshift
            if _mode_ordering != 1:
                raise ValueError(
                    "Double specification of ordering; only one of fftshift and "
                    "modeord should be provided"
                )
            _mode_ordering = 0

        ctx.save_for_backward(points, values)

        ctx.isign = _i_sign
        ctx.mode_ordering = _mode_ordering
        ctx.finufftkwargs = finufftkwargs

        # this below should be a pre-check
        ndim = points.shape[0]
        assert len(output_shape) == ndim

        nufft_func = get_nufft_func(ndim, 1, points.device.type)
        finufft_out = nufft_func(
            *points, values, output_shape, isign=_i_sign, **finufftkwargs
        )
        # because modeord is missing from cufinufft
        if _mode_ordering:
            finufft_out = torch.fft.ifftshift(finufft_out)

        return finufft_out

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
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
        Tuple[Union[torch.Tensor, None], ...]
             A tuple of derivatives wrt. each argument in the forward method
        """
        _i_sign = -1 * ctx.isign
        _mode_ordering = ctx.mode_ordering
        finufftkwargs = ctx.finufftkwargs

        points, values = ctx.saved_tensors

        start_points = -(np.array(grad_output.shape) // 2)
        end_points = start_points + grad_output.shape
        slices = tuple(
            slice(start, end) for start, end in zip(start_points, end_points)
        )

        # CPU idiosyncracy that needs to be done differently
        coord_ramps = torch.from_numpy(np.mgrid[slices]).to(points.device)

        grads_points = None
        grad_values = None

        ndim = points.shape[0]

        nufft_func = get_nufft_func(ndim, 2, points.device.type)

        if ctx.needs_input_grad[0]:
            # wrt points

            if _mode_ordering:
                coord_ramps = torch.fft.ifftshift(
                    coord_ramps, dim=tuple(range(1, ndim + 1))
                )

            ramped_grad_output = coord_ramps * grad_output[np.newaxis] * 1j * _i_sign

            grads_points = []
            for ramp in ramped_grad_output:  # we can batch this into finufft
                if _mode_ordering:
                    ramp = torch.fft.fftshift(ramp)

                backprop_ramp = nufft_func(
                    *points,
                    ramp,
                    isign=_i_sign,
                    **finufftkwargs,
                )

                grad_points = (backprop_ramp.conj() * values).real

                grads_points.append(grad_points)

            grads_points = torch.stack(grads_points)

        if ctx.needs_input_grad[1]:
            if _mode_ordering:
                grad_output = torch.fft.fftshift(grad_output)

            grad_values = nufft_func(
                *points,
                grad_output,
                isign=_i_sign,
                **finufftkwargs,
            )

        return (
            grads_points,
            grad_values,
            None,
            None,
            None,
            None,
        )
