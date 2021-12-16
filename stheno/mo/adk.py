from algebra import WrappedFunction, Function, proven
from algebra.pretty import need_parens
from mlkernels import (
    Kernel,
    pairwise,
    elwise,
)

from .. import _dispatch

__all__ = ["AmbiguousDimensionalityKernel"]


class AmbiguousDimensionalityKernel(Kernel, WrappedFunction):
    """A kernel whose dimensionality cannot be inferred.

    Args:
        k (:class:`mlkernels.Kernel`): Kernel whose dimensionality cannot be inferred.
    """

    def render_wrap(self, e, formatter):
        return str(e)

    @property
    def _stationary(self):
        return self[0].stationary

    @_dispatch
    def __eq__(self, other: "AmbiguousDimensionalityKernel"):
        return self[0] == other[0]


# Will never need parentheses when printing.


@need_parens.dispatch(precedence=proven())
def need_parens(el: Function, parent: AmbiguousDimensionalityKernel):
    return False


@need_parens.dispatch(precedence=proven())
def need_parens(el: AmbiguousDimensionalityKernel, parent: Function):
    return False


# Simply redirect computation.


@pairwise.dispatch
def pairwise(k: AmbiguousDimensionalityKernel, x, y):
    return pairwise(k[0], x, y)


@elwise.dispatch
def elwise(k: AmbiguousDimensionalityKernel, x, y):
    return elwise(k[0], x, y)


# `AmbiguousDimensionalityKernel` cannot infer its dimensionality.


@_dispatch
def dimensionality(k: AmbiguousDimensionalityKernel):
    return None
