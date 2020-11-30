import logging

from lab import B
from plum import Dispatcher, Self, Union

from . import PromisedFDD as FDD
from .input import MultiInput, Input
from .kernel import Kernel

__all__ = ["MultiOutputKernel"]

log = logging.getLogger(__name__)


class MultiOutputKernel(Kernel):
    """A generic multi-output kernel.

    Args:
        measure (:class:`.measure.Measure`): Measure to take the kernels from.
        *ps (:class:`.measure.GP`): Processes that make up the multi-valued process.

    Attributes:
        measure (:class:`.measure.Measure`): Measure to take the kernels from.
        ps (tuple[:class:`.measure.GP`]): Processes that make up the multi-valued
            process.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, measure, *ps):
        self.measure = measure
        self.ps = ps

    # No `FDD` nor `MultiInput`.

    @_dispatch({B.Numeric, Input}, {B.Numeric, Input})
    def __call__(self, x, y):
        return self(
            MultiInput(*(p(x) for p in self.ps)), MultiInput(*(p(y) for p in self.ps))
        )

    # One `FDD`.

    @_dispatch(FDD, {B.Numeric, Input})
    def __call__(self, x, y):
        return self(MultiInput(x), MultiInput(*(p(y) for p in self.ps)))

    @_dispatch({B.Numeric, Input}, FDD)
    def __call__(self, x, y):
        return self(MultiInput(*(p(x) for p in self.ps)), MultiInput(y))

    # Two `FDD`s.

    @_dispatch(FDD, FDD)
    def __call__(self, x, y):
        return self.measure.kernels[x.p, y.p](x.x, y.x)

    # One `MultiInput`.

    @_dispatch(MultiInput, FDD)
    def __call__(self, x, y):
        return self(x, MultiInput(y))

    @_dispatch(MultiInput, {B.Numeric, Input})
    def __call__(self, x, y):
        return self(x, MultiInput(*(p(y) for p in self.ps)))

    @_dispatch(FDD, MultiInput)
    def __call__(self, x, y):
        return self(MultiInput(x), y)

    @_dispatch({B.Numeric, Input}, MultiInput)
    def __call__(self, x, y):
        return self(MultiInput(*(p(x) for p in self.ps)), y)

    # Two `MultiInput`s.

    @_dispatch(MultiInput, MultiInput)
    def __call__(self, x, y):
        return B.block(*[[self(xi, yi) for yi in y.get()] for xi in x.get()])

    # No `FDD` nor `MultiInput`.

    @_dispatch({B.Numeric, Input}, {B.Numeric, Input})
    def elwise(self, x, y):
        return self.elwise(
            MultiInput(*(p(x) for p in self.ps)), MultiInput(*(p(y) for p in self.ps))
        )

    # One `FDD`.

    @_dispatch(FDD, {B.Numeric, Input})
    def elwise(self, x, y):
        raise ValueError(
            "Unclear combination of arguments given to MultiOutputKernel.elwise."
        )

    @_dispatch({B.Numeric, Input}, FDD)
    def elwise(self, x, y):
        raise ValueError(
            "Unclear combination of arguments given to " "MultiOutputKernel.elwise."
        )

    # Two `FDD`s.

    @_dispatch(FDD, FDD)
    def elwise(self, x, y):
        return self.measure.kernels[x.p, y.p].elwise(x.x, y.x)

    # One `MultiInput`.

    @_dispatch(MultiInput, Union(B.Numeric, Input, FDD), precedence=1)
    def elwise(self, x, y):
        raise ValueError(
            "Unclear combination of arguments given to MultiOutputKernel.elwise."
        )

    @_dispatch(Union(B.Numeric, Input, FDD), MultiInput, precedence=1)
    def elwise(self, x, y):
        raise ValueError(
            "Unclear combination of arguments given to MultiOutputKernel.elwise."
        )

    # Two `MultiInput`s.

    @_dispatch(MultiInput, MultiInput)
    def elwise(self, x, y):
        if len(x.get()) != len(y.get()):
            raise ValueError(
                "MultiOutputKernel.elwise must be called with similarly sized "
                "MultiInputs."
            )
        return B.concat(
            *[self.elwise(xi, yi) for xi, yi in zip(x.get(), y.get())], axis=0
        )

    def render(self, formatter):
        ks = [str(self.measure.kernels[p]) for p in self.ps]
        return "MultiOutputKernel({})".format(", ".join(ks))
