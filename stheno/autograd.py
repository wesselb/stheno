# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

from autograd.numpy.numpy_boxes import ArrayBox
from plum import Dispatcher

# noinspection PyUnresolvedReferences
from . import *  # pragma: no cover
from .field import Element
from .random import Random

__all__ = []

log = logging.getLogger(__name__)

_dispatch = Dispatcher()

# Save original methods.
__mul__array_box = ArrayBox.__mul__
__add__array_box = ArrayBox.__add__


@_dispatch(ArrayBox, object)
def __mul__(self, other):
    return __mul__array_box(self, other)


@_dispatch(ArrayBox, {Element, Random})
def __mul__(self, other):
    return other.__rmul__(self)


@_dispatch(ArrayBox, object)
def __add__(self, other):
    return __add__array_box(self, other)


@_dispatch(ArrayBox, {Element, Random})
def __add__(self, other):
    return other.__radd__(self)


# Assign dispatchable methods.
ArrayBox.__mul__ = __mul__
ArrayBox.__add__ = __add__
