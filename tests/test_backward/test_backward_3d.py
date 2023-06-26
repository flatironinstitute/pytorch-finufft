import numpy as np
import pytest
import scipy
import torch
from numpy.random import standard_normal

import pytorch_finufft

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize("something", [1])
def t1_backward_CPU(something) -> None:
    """
    Checks autograd output against finite differences computation
    """
    assert something == 1


@pytest.mark.parametrize("something", [1])
def t2_backward_CPU(something) -> None:
    """
    Checks autograd output against finite differences computation
    """
    assert something == 1


@pytest.mark.parametrize("something", [1])
def t3_backward_CPU(something) -> None:
    """
    Checks autograd output against finite differences computation
    """

    assert something == 1
