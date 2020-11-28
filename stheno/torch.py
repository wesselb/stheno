import logging

# noinspection PyUnresolvedReferences
import lab.torch
from algebra import Element
from matrix import AbstractMatrix
from plum import Dispatcher
from torch import Tensor

# noinspection PyUnresolvedReferences
from . import *  # pragma: no cover
from .random import Random

log = logging.getLogger(__name__)

_dispatch = Dispatcher()

# Save original methods.
__mul__tensor = Tensor.__mul__
__imul__tensor = Tensor.__imul__
__add__tensor = Tensor.__add__
__iadd__tensor = Tensor.__iadd__


@_dispatch(Tensor, object)
def __mul__(self, other):
    return __mul__tensor(self, other)


@_dispatch(Tensor, {Element, Random, AbstractMatrix})
def __mul__(self, other):
    return other.__rmul__(self)


@_dispatch(Tensor, object)
def __imul__(self, other):
    return __imul__tensor(self, other)


@_dispatch(Tensor, {Element, Random, AbstractMatrix})
def __imul__(self, other):
    return other.__rmul__(self)


@_dispatch(Tensor, object)
def __add__(self, other):
    return __add__tensor(self, other)


@_dispatch(Tensor, {Element, Random, AbstractMatrix})
def __add__(self, other):
    return other.__radd__(self)


@_dispatch(Tensor, object)
def __iadd__(self, other):
    return __iadd__tensor(self, other)


@_dispatch(Tensor, {Element, Random, AbstractMatrix})
def __iadd__(self, other):
    return other.__radd__(self)


# Assign dispatchable methods.
Tensor.__mul__ = __mul__
Tensor.__imul__ = __imul__
Tensor.__add__ = __add__
Tensor.__iadd__ = __iadd__
