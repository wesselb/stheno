import logging

from lab import B
from plum import Dispatcher, Union

from . import PromisedFDD as FDD
from .input import MultiInput, Input
from .kernel import Kernel

__all__ = ["MultiOutputKernel"]

_dispatch = Dispatcher()

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

    def __init__(self, measure, *ps):
        self.measure = measure
        self.ps = ps

    # No `FDD` nor `MultiInput`.

    @_dispatch
    def __call__(self, x: Union[B.Numeric, Input], y: Union[B.Numeric, Input]):
        return self(
            MultiInput(*(p(x) for p in self.ps)), MultiInput(*(p(y) for p in self.ps))
        )

    # One `FDD`.

    @_dispatch
    def __call__(self, x: FDD, y: Union[B.Numeric, Input]):
        return self(MultiInput(x), MultiInput(*(p(y) for p in self.ps)))

    @_dispatch
    def __call__(self, x: Union[B.Numeric, Input], y: FDD):
        return self(MultiInput(*(p(x) for p in self.ps)), MultiInput(y))

    # Two `FDD`s.

    @_dispatch
    def __call__(self, x: FDD, y: FDD):
        return self.measure.kernels[x.p, y.p](x.x, y.x)

    # One `MultiInput`.

    @_dispatch
    def __call__(self, x: MultiInput, y: FDD):
        return self(x, MultiInput(y))

    @_dispatch
    def __call__(self, x: MultiInput, y: Union[B.Numeric, Input]):
        return self(x, MultiInput(*(p(y) for p in self.ps)))

    @_dispatch
    def __call__(self, x: FDD, y: MultiInput):
        return self(MultiInput(x), y)

    @_dispatch
    def __call__(self, x: Union[B.Numeric, Input], y: MultiInput):
        return self(MultiInput(*(p(x) for p in self.ps)), y)

    # Two `MultiInput`s.

    @_dispatch
    def __call__(self, x: MultiInput, y: MultiInput):
        return B.block(*[[self(xi, yi) for yi in y.get()] for xi in x.get()])

    # No `FDD` nor `MultiInput`.

    @_dispatch
    def elwise(self, x: Union[B.Numeric, Input], y: Union[B.Numeric, Input]):
        return self.elwise(
            MultiInput(*(p(x) for p in self.ps)), MultiInput(*(p(y) for p in self.ps))
        )

    # One `FDD`.

    @_dispatch
    def elwise(self, x: FDD, y: Union[B.Numeric, Input]):
        raise ValueError(
            "Unclear combination of arguments given to MultiOutputKernel.elwise."
        )

    @_dispatch
    def elwise(self, x: Union[B.Numeric, Input], y: FDD):
        raise ValueError(
            "Unclear combination of arguments given to " "MultiOutputKernel.elwise."
        )

    # Two `FDD`s.

    @_dispatch
    def elwise(self, x: FDD, y: FDD):
        return self.measure.kernels[x.p, y.p].elwise(x.x, y.x)

    # One `MultiInput`.

    @_dispatch(precedence=1)
    def elwise(self, x: MultiInput, y: Union[B.Numeric, Input, FDD]):
        raise ValueError(
            "Unclear combination of arguments given to MultiOutputKernel.elwise."
        )

    @_dispatch(precedence=1)
    def elwise(self, x: Union[B.Numeric, Input, FDD], y: MultiInput):
        raise ValueError(
            "Unclear combination of arguments given to MultiOutputKernel.elwise."
        )

    # Two `MultiInput`s.

    @_dispatch
    def elwise(self, x: MultiInput, y: MultiInput):
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
