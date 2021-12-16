import lab as B
from algebra import Element, Wrapped, Join
from lab.shape import Dimension
from mlkernels import (
    Kernel,
    num_elements,
    PosteriorKernel,
    SubspaceKernel,
)

from .. import _dispatch, PromisedFDD as FDD

__all__ = ["infer_size", "dimensionality"]


@num_elements.dispatch
def num_elements(x: tuple):
    return sum(map(num_elements, x))


@_dispatch
def infer_size(k: Kernel, x: tuple):
    """Infer the size of `k` evaluated at `x`.

    Args:
        k (:class:`mlkernels.Kernel`): Kernel to evaluate.
        x (input): Input to evaluate kernel at.

    Returns:
        int: Size of kernel matrix.
    """
    return Dimension(sum([infer_size(k, xi) for xi in x]))


@_dispatch
def infer_size(k: Kernel, x: B.Numeric):
    d = dimensionality(k)
    if d is None:
        raise RuntimeError(f"Could not infer dimensionality of {k}.")
    return Dimension(num_elements(x) * d)


@_dispatch
def infer_size(k: Kernel, x: FDD):
    return Dimension(num_elements(x))


def _check_and_merge(k, *ds):
    ds = list(filter(None, ds))
    if len(ds) == 0:
        return None
    if not all([d == ds[0] for d in ds[1:]]):
        raise RuntimeError(f"Inferred dimensionalities for kernel {k} do not match. ")
    return ds[0]


@_dispatch
def dimensionality(k: Join):
    """Infer the output dimensionality of `k`.

    Args:
        k (:class:`mlkernels.Kernel`): Kernel to get the output dimensionality of.

    Returns:
        int: Output dimensionality of `k`.
    """
    d1 = dimensionality(k[0])
    d2 = dimensionality(k[1])
    return _check_and_merge(k, d1, d2)


@_dispatch
def dimensionality(k: Wrapped):
    return dimensionality(k[0])


@_dispatch
def dimensionality(k: Element):
    return 1


# `PosteriorKernel` and `SubspaceKernel` are not `Join` nor `Wrapped`, so we must
# implement methods for them. TODO: Automatically detect these kernels via a type?


@_dispatch
def dimensionality(k: PosteriorKernel):
    return _check_and_merge(
        k,
        dimensionality(k.k_ij),
        dimensionality(k.k_zi),
        dimensionality(k.k_zj),
    )


@_dispatch
def dimensionality(k: SubspaceKernel):
    return _check_and_merge(
        k,
        dimensionality(k.k_zi),
        dimensionality(k.k_zj),
    )
