import logging

# noinspection PyUnresolvedReferences
import lab.tensorflow
from algebra import Element
from matrix import AbstractMatrix
from plum import Dispatcher
from tensorflow import Tensor, Variable

# noinspection PyUnresolvedReferences
from . import *  # pragma: no cover
from .random import Random

log = logging.getLogger(__name__)

_dispatch = Dispatcher()

# Save original methods.
__mul__tensor = Tensor.__mul__
__mul__variable = Variable.__mul__

__add__tensor = Tensor.__add__
__add__variable = Variable.__add__


@_dispatch(Tensor, object)
def __mul__(self, other):
    return __mul__tensor(self, other)


@_dispatch(Variable, object)
def __mul__(self, other):
    return __add__variable(self, other)


@_dispatch.multi(
    (Tensor, {Element, Random, AbstractMatrix}),
    (Variable, {Element, Random, AbstractMatrix}),
)
def __mul__(self, other):
    return other.__rmul__(self)


@_dispatch(Tensor, object)
def __add__(self, other):
    return __add__tensor(self, other)


@_dispatch(Variable, object)
def __add__(self, other):
    return __add__variable(self, other)


@_dispatch.multi(
    (Tensor, {Element, Random, AbstractMatrix}),
    (Variable, {Element, Random, AbstractMatrix}),
)
def __add__(self, other):
    return other.__radd__(self)


# Assign dispatchable methods.
Tensor.__mul__ = __mul__
Variable.__mul__ = __mul__

Tensor.__add__ = __add__
Variable.__add__ = __add__
