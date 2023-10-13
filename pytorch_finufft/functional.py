"""
Implementations of the corresponding Autograd functions
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

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


class finufft_type2(torch.autograd.Function):
    """
    FINUFFT 2D problem type 2
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        points: torch.Tensor,
        targets: torch.Tensor,
        out: Optional[torch.Tensor] = None,
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

        if out is not None:
            print("In-place results are not yet implemented")

        # TODO -- extend checks to 2d
        checks._type2_checks(points, targets)

        if finufftkwargs is None:
            finufftkwargs = dict()

        finufftkwargs = {k: v for k, v in finufftkwargs.items()}
        _mode_ordering = finufftkwargs.pop(
            "modeord", 1
        )  # not finufft default, but corresponds to pytorch default
        _i_sign = finufftkwargs.pop(
            "isign", -1
        )  # isign=-1 is finufft default for type 2

        ndim = points.shape[0]
        if _mode_ordering == 1:
            targets = torch.fft.fftshift(targets)

        ctx.isign = _i_sign
        ctx.mode_ordering = _mode_ordering
        ctx.finufftkwargs = finufftkwargs

        ctx.save_for_backward(points, targets)

        nufft_func = get_nufft_func(ndim, 2, points.device.type)

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

            if _mode_ordering == 1:
                grad_targets = torch.fft.ifftshift(grad_targets)

        return (
            grad_points,
            grad_targets,
            None,
            None,
            None,
        )
