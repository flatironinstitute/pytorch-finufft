"""
Implementations of the corresponding Autograd functions
"""

import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

try:
    import finufft

    FINUFFT_AVAIL = True
except ImportError:
    FINUFFT_AVAIL = False

try:
    import cufinufft

    if cufinufft.__version__.startswith("1."):
        warnings.warn("pytorch-finufft does not support cufinufft v1.x.x")
    else:
        CUFINUFFT_AVAIL = True
except ImportError:
    CUFINUFFT_AVAIL = False

if not (FINUFFT_AVAIL or CUFINUFFT_AVAIL):
    raise ImportError(
        "No FINUFFT implementation available. "
        "Install either finufft or cufinufft and ensure they are importable."
    )

import pytorch_finufft.checks as checks

newaxis = None


def get_nufft_func(
    dim: int, nufft_type: int, device_type: str
) -> Callable[..., torch.Tensor]:
    if device_type == "cuda":
        if not CUFINUFFT_AVAIL:
            raise RuntimeError("CUDA device requested but cufinufft failed to import")
        return getattr(cufinufft, f"nufft{dim}d{nufft_type}")  # type: ignore

    if not FINUFFT_AVAIL:
        raise RuntimeError("CPU device requested but finufft failed to import")
    # CPU needs extra work to go to/from torch and numpy
    finufft_func = getattr(finufft, f"nufft{dim}d{nufft_type}")

    def f(*args, **kwargs):
        new_args = [arg for arg in args]
        for i in range(len(new_args)):
            if isinstance(new_args[i], torch.Tensor):
                new_args[i] = new_args[i].data.numpy()

        return torch.from_numpy(finufft_func(*new_args, **kwargs))

    return f


def coordinate_ramps(shape, device):
    start_points = -(torch.tensor(shape, device=device) // 2)
    end_points = start_points + torch.tensor(shape, device=device)
    coord_ramps = torch.stack(
        torch.meshgrid(
            *(
                torch.arange(start, end, device=device)
                for start, end in zip(start_points, end_points)
            ),
            indexing="ij",
        )
    )

    return coord_ramps


class FinufftType1(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        points: torch.Tensor,
        values: torch.Tensor,
        output_shape: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
        finufftkwargs: Optional[Dict[str, Union[int, float]]] = None,
    ) -> torch.Tensor:
        """
        Evaluates the Type 1 NUFFT on the inputs.
        """

        checks.check_devices(values, points)
        checks.check_dtypes(values, points, "Values")
        checks.check_sizes_t1(values, points)
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
            coord_ramps = coordinate_ramps(grad_output.shape, device)

            # we can't batch in 1d case so we squeeze and fix up the ouput later
            ramped_grad_output = (
                coord_ramps * grad_output[newaxis] * 1j * _i_sign
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


class FinufftType2(torch.autograd.Function):
    """
    FINUFFT 2D problem type 2
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        points: torch.Tensor,
        targets: torch.Tensor,
        finufftkwargs: Optional[Dict[str, Union[int, float]]] = None,
    ) -> torch.Tensor:
        """
        Evaluates the Type 2 NUFFT on the inputs.

        NOTE: By default, the ordering is set to match that of Pytorch,
         Numpy, and Scipy's FFT APIs. To match the mode ordering
         native to FINUFFT, add {'modeord': 0} to finufftkwargs.

        Parameters
        ----------
        ctx : Any
            Pytorch context objecy
        points : torch.Tensor, shape=(ndim, num_points)
            The non-uniform points x
        targets : torch.Tensor
            The values on the input grid
        out : Optional[torch.Tensor], optional
            Array to take the result in-place, by default None
        finufftkwargs : Dict[str, Union[int, float]]
            Additional arguments will be passed into FINUFFT. See
            https://finufft.readthedocs.io/en/latest/python.html.

        Returns
        -------
        torch.Tensor
            The Fourier transform of the targets grid evaluated at the points `points`

        Raises
        ------

        """
        checks.check_devices(targets, points)
        checks.check_dtypes(targets, points, "Targets")
        checks.check_sizes_t2(targets, points)

        if finufftkwargs is None:
            finufftkwargs = dict()
        finufftkwargs = {k: v for k, v in finufftkwargs.items()}
        _mode_ordering = finufftkwargs.pop(
            "modeord", 1
        )  # not finufft default, but corresponds to pytorch default
        _i_sign = finufftkwargs.pop(
            "isign", -1
        )  # isign=-1 is finufft default for type 2

        points = torch.atleast_2d(points)
        if _mode_ordering:
            targets = torch.fft.fftshift(targets)

        ctx.save_for_backward(points, targets)

        ctx.isign = _i_sign
        ctx.mode_ordering = _mode_ordering
        ctx.finufftkwargs = finufftkwargs

        nufft_func = get_nufft_func(points.shape[0], 2, points.device.type)

        finufft_out = nufft_func(
            *points,
            targets,
            isign=_i_sign,
            **finufftkwargs,
        )

        return finufft_out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], None, None, None,]:
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

        points, targets = ctx.saved_tensors
        device = points.device

        grad_points = grad_targets = None
        ndim = points.shape[0]

        if ctx.needs_input_grad[0]:
            coord_ramps = coordinate_ramps(targets.shape, device=device)
            ramped_targets = coord_ramps * targets[newaxis] * 1j * _i_sign
            nufft_func = get_nufft_func(ndim, 2, points.device.type)

            grad_points = nufft_func(
                *points,
                ramped_targets.squeeze(),
                isign=_i_sign,
                **finufftkwargs,
            ).conj()  # Why can't this be replaced with a flipped isign

            grad_points = grad_points * grad_output
            grad_points = torch.atleast_2d(grad_points.real)

        if ctx.needs_input_grad[1]:
            # wrt. targets
            nufft_func = get_nufft_func(ndim, 1, points.device.type)

            grad_targets = nufft_func(
                *points,
                grad_output,
                targets.shape,
                isign=-_i_sign,
                **finufftkwargs,
            )

            if _mode_ordering:
                grad_targets = torch.fft.ifftshift(grad_targets)

        return (
            grad_points,
            grad_targets,
            None,
            None,
            None,
        )


def finufft_type1(
    points: torch.Tensor,
    values: torch.Tensor,
    output_shape: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    **finufftkwargs: Union[int, float],
) -> torch.Tensor:
    """
    Evaluates the Type 1 (nonuniform-to-uniform) NUFFT on the inputs.

    This is a wrapper around :func:`finufft.nufft1d1`, :func:`finufft.nufft2d1`, and
    :func:`finufft.nufft3d1` on CPU, and :func:`cufinufft.nufft1d1`,
    :func:`cufinufft.nufft2d1`, and :func:`cufinufft.nufft3d1` on GPU.

    Parameters
    ----------
    points : torch.Tensor
        DxN tensor of locations of the non-uniform points.
        Points must lie in the range ``[-3pi, 3pi]``.
    values : torch.Tensor
        Length N complex-valued tensor of values at the non-uniform points
    output_shape : int | tuple(int, ...)
        Requested output shape of Fourier modes. Must be a tuple of length D or
        an integer (1D only).
    **finufftkwargs : int | float
        Additional keyword arguments are forwarded to the underlying
        FINUFFT functions. A few notable options are

        - ``eps``: precision requested (default: ``1e-6``)
        - ``modeord``: 0 for FINUFFT default, 1 for Pytorch default (default: ``1``)
        - ``isign``: Sign of the exponent in the Fourier transform (default: ``-1``)

    Returns
    -------
    torch.Tensor
        Tensor with shape ``output_shape`` containing the Fourier
        transform of the values.
    """
    res: torch.Tensor = FinufftType1.apply(points, values, output_shape, finufftkwargs)
    return res


def finufft_type2(
    points: torch.Tensor,
    targets: torch.Tensor,
    **finufftkwargs: Union[int, float],
) -> torch.Tensor:
    """
    Evaluates the Type 2 (uniform-to-nonuniform) NUFFT on the inputs.

    This is a wrapper around :func:`finufft.nufft1d2`, :func:`finufft.nufft2d2`, and
    :func:`finufft.nufft3d2` on CPU, and :func:`cufinufft.nufft1d2`,
    :func:`cufinufft.nufft2d2`, and :func:`cufinufft.nufft3d2` on GPU.

    Parameters
    ----------
    points : torch.Tensor
        DxN tensor of locations of the non-uniform points.
        Points must lie in the range ``[-3pi, 3pi]``.
    targets : torch.Tensor
        D-dimensional complex-valued tensor of Fourier modes to evaluate at the points
    **finufftkwargs : int | float
        Additional keyword arguments are forwarded to the underlying
        FINUFFT functions. A few notable options are

        - ``eps``: precision requested (default: ``1e-6``)
        - ``modeord``: 0 for FINUFFT default, 1 for Pytorch default (default: ``1``)
        - ``isign``: Sign of the exponent in the Fourier transform (default: ``-1``)

    Returns
    -------
    torch.Tensor
        A DxN tensor of values at the non-uniform points.
    """
    res: torch.Tensor = FinufftType2.apply(points, targets, finufftkwargs)
    return res
