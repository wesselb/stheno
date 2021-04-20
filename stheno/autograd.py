import logging
from typing import Union

# noinspection PyUnresolvedReferences
import lab.autograd as B
from algebra import Element
from autograd.numpy.numpy_boxes import ArrayBox
from matrix import AbstractMatrix
from plum import Dispatcher

# noinspection PyUnresolvedReferences
from . import *  # pragma: no cover
from .random import Random

__all__ = []

log = logging.getLogger(__name__)

_dispatch = Dispatcher()

# Save original methods.
__mul__array_box = ArrayBox.__mul__
__add__array_box = ArrayBox.__add__


@_dispatch
def __mul__(self: ArrayBox, other):
    return __mul__array_box(self, other)


@_dispatch
def __mul__(self: ArrayBox, other: Union[Element, Random, AbstractMatrix]):
    return other.__rmul__(self)


@_dispatch
def __add__(self: ArrayBox, other):
    return __add__array_box(self, other)


@_dispatch
def __add__(self: ArrayBox, other: Union[Element, Random, AbstractMatrix]):
    return other.__radd__(self)


# Assign dispatchable methods.
ArrayBox.__mul__ = __mul__
ArrayBox.__add__ = __add__
