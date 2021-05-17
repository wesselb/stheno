from lab import B
from matrix import AbstractMatrix, Dense, Zero, Diagonal
from mlkernels import num_elements
from wbml.util import indented_kv

from .. import PromisedFDD, PromisedGP, _dispatch
from ..mo import infer_size
from ..random import Normal

__all__ = ["FDD"]


@_dispatch
def _noise_as_matrix(noise: type(None), dtype: B.DType, n: int):
    """Efficiently represent noise as a matrix.

    Args:
        noise (None, scalar, vector, or matrix): Noise. `None` means no noise.
        dtype (dtype): Data type that the noise should be.
        n (int): Number of observations.

    Returns:
        matrix: Noise as a matrix.
    """
    return Zero(dtype, n, n)


@_dispatch
def _noise_as_matrix(noise: B.Numeric, dtype: B.DType, n: int):
    if B.isscalar(noise):
        return B.fill_diag(noise, n)
    elif B.rank(noise) == 1:
        return Diagonal(noise)
    else:
        return Dense(noise)


@_dispatch
def _noise_as_matrix(noise: AbstractMatrix, dtype: B.DType, n: int):
    return noise


class FDD(Normal):
    """Finite-dimensional distribution.

    Args:
        p (:class:`stheno.random.RandomProcess` or int): Process of FDD. Can also be the
            `id` of the process.
        x (input): Inputs that `p` is evaluated at.
        noise (scalar, vector, or matrix, optional): Additive noise.

    Attributes:
        p (:class:`stheno.random.RandomProcess` or int): Process of FDD.
        x (input): Inputs that `p` is evaluated at.
        noise (matrix or None): Additive noise. `None` if `p` is an `int`.
    """

    @_dispatch
    def __init__(self, p: PromisedGP, x, noise):
        self.p = p
        self.x = x
        self.noise = _noise_as_matrix(noise, B.dtype(x), infer_size(p.kernel, x))
        Normal.__init__(self, lambda: p.mean(x), lambda: p.kernel(x) + self.noise)

    @_dispatch
    def __init__(self, p: PromisedGP, x):
        FDD.__init__(self, p, x, None)

    @_dispatch
    def __init__(self, p: int, x):
        self.p = p
        self.x = x
        self.noise = None

    def __str__(self):
        return (
            f"<FDD:\n"
            + indented_kv("process", str(self.p), suffix=",\n")
            + indented_kv("input", str(self.x), suffix=",\n")
            + indented_kv("noise", str(self.noise), suffix=">")
        )

    def __repr__(self):
        return (
            f"<FDD:\n"
            + indented_kv("process", repr(self.p), suffix=",\n")
            + indented_kv("input", repr(self.x), suffix=",\n")
            + indented_kv("noise", repr(self.noise), suffix=">")
        )


PromisedFDD.deliver(FDD)


@B.dtype.dispatch
def dtype(fdd: FDD):
    return B.dtype(fdd.x)


@num_elements.dispatch
def num_elements(fdd: FDD):
    return num_elements(fdd.x)
