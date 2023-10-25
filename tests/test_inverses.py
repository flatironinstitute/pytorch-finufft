import numpy as np
import pytest
import torch

import pytorch_finufft

Ns = [
    5,
    10,
    15,
    100,
]

dims = [1, 2, 3]


def check_t2_ifft_undoes_t1(N: int, dim: int, device: str) -> None:
    """
    Tests that nuifft_type2 undoes nufft_type1
    """
    slices = tuple(slice(None, N) for _ in range(dim))
    g = np.mgrid[slices] * 2 * np.pi / N
    points = torch.from_numpy(g.reshape(dim, -1)).to(device)

    # batched values to test that functionality for these as well
    values = torch.randn(3, *points[0].shape, dtype=torch.complex128).to(device)

    print("N is " + str(N))
    print("dim is " + str(dim))
    print("shape of points is " + str(points.shape))
    print("shape of values is " + str(values.shape))

    finufft_out = pytorch_finufft.functional.finufft_type1(
        points,
        values,
        tuple(N for _ in range(dim)),
    )

    back = pytorch_finufft.functional.finuifft_type2(
        points,
        finufft_out,
    )

    np.testing.assert_allclose(values.cpu().numpy(), back.cpu().numpy(), atol=1e-4)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("dim", dims)
def test_t2_ifft_undoes_t1_forward_CPU(N, dim):
    check_t2_ifft_undoes_t1(N, dim, "cpu")


def check_t1_ifft_undoes_t2(N: int, dim: int, device: str) -> None:
    """
    Tests that nuifft_type1 undoes nufft_type2
    """
    slices = tuple(slice(None, N) for _ in range(dim))
    g = np.mgrid[slices] * 2 * np.pi / N
    points = torch.from_numpy(g.reshape(g.shape[0], -1)).to(device)

    # batched targets to test that functionality for these as well
    targets = torch.randn(3, *g[0].shape, dtype=torch.complex128).to(device)

    print("N is " + str(N))
    print("dim is " + str(dim))
    print("shape of points is " + str(points.shape))
    print("shape of targets is " + str(targets.shape))

    finufft_out = pytorch_finufft.functional.finufft_type2(
        points,
        targets,
    )

    back = pytorch_finufft.functional.finuifft_type1(
        points,
        finufft_out,
        tuple(N for _ in range(dim)),
    )

    np.testing.assert_allclose(targets.cpu().numpy(), back.cpu().numpy(), atol=1e-4)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("dim", dims)
def test_t1_ifft_undoes_t2_forward_CPU(N, dim):
    check_t1_ifft_undoes_t2(N, dim, "cpu")
