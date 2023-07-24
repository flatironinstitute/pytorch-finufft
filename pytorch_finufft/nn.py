from typing import Optional

import torch

import pytorch_finufft.functional as func


class Finufft1D1(torch.nn.Module):
    def __init__(
        self,
    ):
        """
        TODO
        """
        super().__init__()

    def forward(
        self,
        points: torch.Tensor,
        values: torch.Tensor,
        output_shape: int,
        out: Optional[torch.Tensor] = None,
        fftshift: bool = False,
        **finufftkwargs: Optional[str],
    ) -> torch.Tensor:
        """
        Evalutes the Type 1 NUFFT on the inputs.

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
        """
        return func.finufft1D1.apply(
            points, values, output_shape, out, fftshift, **finufftkwargs
        )


class Finufft1D2(torch.nn.Module):
    def __init__(
        self,
    ):
        """
        TODO
        """
        super().__init__()

        # TODO

    def forward(
        self,
        points: torch.Tensor,
        values: torch.Tensor,
        output_shape: int,
        out: Optional[torch.Tensor] = None,
        fftshift: bool = False,
        **finufftkwargs: Optional[str],
    ) -> torch.Tensor:
        """
        Evalutes the Type 1 NUFFT on the inputs.

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
        points : torch.Tensor
            The non-uniform points x_j. Valid only between -3pi and 3pi.
        values : torch.Tensor
            The source strengths c_j.
        output_shape : int
            Number of Fourier modes to use in the computation
        out : Optional[torch.Tensor], optional
            _description_, by default None
        fftshift : bool, optional
            _description_, by default False

        Returns
        -------
        torch.Tensor
            _description_
        """
        return func.finufft1D2.apply(
            points, values, output_shape, out, fftshift, **finufftkwargs
        )
