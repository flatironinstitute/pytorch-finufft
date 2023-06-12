"""
Implementations of the corresponding Autograd functions
"""

from typing import Union

import finufft
import torch


class finufft1D1(torch.autograd.Function):
    """
    FINUFFT 1D problem type 1 (non-uniform points)
    """

    @staticmethod
    def forward(
        points: torch.Tensor,
        values: torch.Tensor,
        output_shape: Union[int, None] = None,
        out: Union[torch.Tensor, None] = None,
        fftshift=False,
        **finufftkwargs,
    ) -> torch.Tensor:
        """Evaluates the Type 1 NUFFT on the inputs.

        NOTE: By default here, the ordering is set to match that of Pytorch,
         Numpy, and Scipy's FFT implementations. To match the mode ordering
         native to FINUFFT, set `fftshift = True`
        ```
                M-1
        f[k1] = SUM c[j] exp(+/-i k1 x(j))
                j=0

            for -N1/2 <= k1 <= (N1-1)/2
        ```

        Args:
            points (torch.Tensor): The non-uniform points `x_j`; valid only
                between -3pi and 3pi.
            values (torch.Tensor): The source strengths `c_j`
            output_shape: Number of Fourier modes to use in the computation;
                should be specified if `out` is not given.
            out: Vector to fill in-place with resulting values; should be
                provided if `output_shape` is not given
            fftshift (bool): If true, centers the 0 mode in the
                resultant `torch.Tensor`
            **finufftkwargs: Keyword arguments to be passed directly
                into FINUFFT Python API

        Returns:
            torch.Tensor(complex[N1] or complex[ntransf, N1]): The
                resultant array

        Raises:
            ValueError: If out TODO
            ValueError: If either points or values are not torch.Tensor's
        """

        if output_shape is None:
            output_shape = len(points)

        # TODO: restore total API

        finufft_out = finufft.nufft1d1(
            points.numpy(), values.numpy(), output_shape, modeord=1, isign=-1
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
        out: Union[torch.Tensor, None] = None,
        fftshift: bool = False,
        isign: int = 1,
        **finufftkwargs,
    ) -> torch.Tensor:
        """Evaluates Type 2 NUFFT on inputs

        ```
        c[j] = SUM f[k1] exp(+/-i k1 x(j))
               k1

            for j = 0, ..., M-1, where the sum is over -N1/2 <= k1 <= (N1-1)/2
        ```

        Args:
            points: The non-uniform points `x_j`; valid only between -3pi and 3pi
            targets: Fourier mode coefficient tensor of length N1, where N1 may be even or odd.
            out: Array to take the output in-place
            fftshift: If true, centers the 0 mode in the resultant `torch.Tensor`
            **finufftkwargs: Keyword arguments
                # TODO -- link the one FINUFFT page regarding keywords

        Raises:
            ValueError: Since `out` is not yet implemented but want to keep function signature
            ValueError: If either points or targets is not a torch.Tensor

        Returns:
            torch.Tensor(complex[M] or complex[ntransf, M]): The resulting array
        """

        if not (
            isinstance(points, torch.Tensor)
            and isinstance(targets, torch.Tensor)
        ):
            raise TypeError("Both `points` and `targets` must be torch.Tensor")

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
        out: Union[torch.Tensor, None] = None,
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

        Raises:
            ValueError: If `out` b/c TBD

        Returns:
            torch.Tensor(complex[M] or complex[ntransf, M]): The resulting array
        """

        if out is not None:
            raise ValueError("In-place not implemented")

        isign = finufftkwargs.get("isign") if "isign" in finufftkwargs else -1
        modeord = 0 if fftshift else 1

        # TODO -- size checks and so on for the tensors; finufft will handle the rest of these

        nufft_out = finufft.nufft1d3(
            points.numpy(),
            values.numpy(),
            targets.numpy(),
            isign=isign,
            modeord=modeord,
            **finufftkwargs,
        )

        return torch.from_numpy(nufft_out)

    @staticmethod
    def setup_context(_):
        raise ValueError("TBD")

    @staticmethod
    def backward(_):
        raise ValueError("TBD")
