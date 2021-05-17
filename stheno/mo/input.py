import lab as B
from algebra import Element, Join, Wrapped
from mlkernels import pairwise, elwise, num_elements, Kernel

from .. import PromisedFDD as FDD, _dispatch

__all__ = ["infer_size", "dimensionality"]


@pairwise.dispatch(precedence=1)
def pairwise(k, x: tuple, y: tuple):
    return B.block(*[[pairwise(k, xi, yi) for yi in y] for xi in x])


@pairwise.dispatch(precedence=1)
def pairwise(k, x: tuple, y):
    return pairwise(k, x, (y,))


@pairwise.dispatch(precedence=1)
def pairwise(k, x, y: tuple):
    return pairwise(k, (x,), y)


@elwise.dispatch(precedence=1)
def elwise(k, x: tuple, y: tuple):
    if len(x) != len(y):
        raise ValueError('"elwise" must be called with similarly sized tuples.')
    return B.concat(*[elwise(k, xi, yi) for xi, yi in zip(x, y)], axis=0)


@elwise.dispatch(precedence=1)
def elwise(k, x: tuple, y):
    return elwise(k, x, (y,))


@elwise.dispatch(precedence=1)
def elwise(k, x, y: tuple):
    return elwise(k, (x,), y)


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
    return sum([infer_size(k, xi) for xi in x])


@_dispatch
def infer_size(k: Kernel, x: B.Numeric):
    return num_elements(x) * dimensionality(k)


@_dispatch
def infer_size(k: Kernel, x: FDD):
    return num_elements(x)


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
    if d1 != d2:
        raise RuntimeError(
            f"Inferred dimensionalities {d1} and {d2} do not match. "
            f"Did you join incompatible elements?"
        )
    return d1


@_dispatch
def dimensionality(k: Wrapped):
    return dimensionality(k[0])


@_dispatch
def dimensionality(k: Element):
    return 1
