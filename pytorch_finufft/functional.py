"""
Implementations of the corresponding Autograd functions
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

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

import pytorch_finufft.checks as checks

###############################################################################
# 1d Functions
###############################################################################


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

        checks._type2_checks((points,), targets)

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


###############################################################################
# 2d Functions
###############################################################################


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
        checks._type2_checks((points_x, points_y), targets)

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


###############################################################################
# 3d Functions
###############################################################################


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

        checks._type2_checks((points_x, points_y, points_z), targets)

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


###############################################################################
# Consolidated forward function for all 1D, 2D, and 3D problems for nufft type 1
###############################################################################


def get_nufft_func(
    dim: int, nufft_type: int, device_type: str
) -> Callable[..., torch.Tensor]:
    if device_type == "cuda":
        return getattr(cufinufft, f"nufft{dim}d{nufft_type}")  # type: ignore

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
    def forward(  # type: ignore[override]
        ctx: Any,
        points: torch.Tensor,
        values: torch.Tensor,
        output_shape: Union[int, Tuple[int, int], Tuple[int, int, int]],
        out: Optional[torch.Tensor] = None,
        finufftkwargs: Optional[Dict[str, Union[int, float]]] = None,
    ) -> torch.Tensor:
        """
        Evaluates the Type 1 NUFFT on the inputs.
        """

        if out is not None:
            # All this requires is a check on the out array to make sure it is the
            # correct shape.
            raise NotImplementedError("In-place results are not yet implemented")

        checks.check_devices(values, points)
        checks.check_dtypes(values, points)
        checks.check_sizes(values, points)
        points = torch.atleast_2d(points)
        ndim = points.shape[0]
        checks.check_output_shape(ndim, output_shape)

        ctx.save_for_backward(points, values)

        if finufftkwargs is None:
            finufftkwargs = {}
        else:  # copy to avoid mutating caller's dictionary
            finufftkwargs = {k: v for k, v in finufftkwargs.items()}
        ctx.isign = finufftkwargs.pop("isign", -1)  # note: FINUFFT default is 1
        ctx.mode_ordering = finufftkwargs.pop(
            "modeord", 1
        )  # note: FINUFFT default is 0
        ctx.finufftkwargs = finufftkwargs

        nufft_func = get_nufft_func(ndim, 1, points.device.type)
        finufft_out = nufft_func(
            *points, values, output_shape, isign=ctx.isign, **ctx.finufftkwargs
        )

        # because modeord is missing from cufinufft
        if ctx.mode_ordering:
            finufft_out = torch.fft.ifftshift(finufft_out)

        return finufft_out

    @staticmethod
    def backward(  # type: ignore[override]
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
        device = points.device

        grads_points = None
        grad_values = None

        ndim = points.shape[0]

        nufft_func = get_nufft_func(ndim, 2, device.type)

        if any(ctx.needs_input_grad) and _mode_ordering:
            grad_output = torch.fft.fftshift(grad_output)

        if ctx.needs_input_grad[0]:
            # wrt points
            start_points = -(torch.tensor(grad_output.shape, device=device) // 2)
            end_points = start_points + torch.tensor(grad_output.shape, device=device)
            coord_ramps = torch.stack(
                torch.meshgrid(
                    *(
                        torch.arange(start, end, device=device)
                        for start, end in zip(start_points, end_points)
                    ),
                    indexing="ij",
                )
            )

            # we can't batch in 1d case so we squeeze and fix up the ouput later
            ramped_grad_output = (
                coord_ramps * grad_output[np.newaxis] * 1j * _i_sign
            ).squeeze()
            backprop_ramp = nufft_func(
                *points, ramped_grad_output, isign=_i_sign, **finufftkwargs
            )
            grads_points = torch.atleast_2d((backprop_ramp.conj() * values).real)

        if ctx.needs_input_grad[1]:
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
