"""
Implementations of the corresponding Autograd functions
"""

from typing import Any, Literal

import finufft
import torch


class finufft1D1(torch.autograd.Function):
    """
    FINUFFT 1D problem type 1 (non-uniform points)
    """

    @staticmethod
    def forward(
        x, c, n_modes=None, out=None, eps=1e-06, isign=-1, modeord=1, **kwargs
    ) -> torch.Tensor:
        """
        Forward method, returns the `finufft.nufft1d1` call on the inputs, ie, the 1D type-1 (nonuniform to uniform)
        complex NUFFT.

        NOTE: By default here, the ordering is set to match that of Pytorch, Numpy, and Scipy's FFT implementations.
            By default in FINUFFT, `modeord` would be set to `0` instead.
        ```
                M-1
        f[k1] = SUM c[j] exp(+/-i k1 x(j))
                j=0

            for -N1/2 <= k1 <= (N1-1)/2
        ```

        Args:
            x (float[M]): nonuniform points, valid only in [-3pi, 3pi]
            c (complex[M]): source strengths per point in `x`
            n_modes (integer, optional): number of uniform Fourier modes requested. May be even or odd
            out (complex[N1] or complex[ntransf, N1], optional): output array for Fourier mode values.
                If `n_modes`

            isign (int, optional): if non-negative, uses positive sign in exponential, otherwise negative sign.
            modeord (int, optional): if `0`, frequency indices are in increasing ordering, otherwise frequency
                indices are ordered as in the usual FFT, increasing from zero then jumping to negative indices
                halfway along.

        Returns:
            torch.Tensor(complex[N1] or complex[ntransf, N1]): The resulting array

        """

        if n_modes is None and out is None:
            n_modes = len(c)

        return torch.from_numpy(
            finufft.nufft1d1(
                x, c, n_modes, out, eps, isign=isign, modeord=modeord, **kwargs
            )
        )

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        return 0

    @staticmethod
    def backward(ctx, grad_output):
        """
        Implements gradients for backward mode automatic differentiation
        """

        return 0
