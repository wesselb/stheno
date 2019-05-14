# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import operator
from types import FunctionType as PythonFunction
import numpy as np

from lab import B
from plum import Dispatcher, Referentiable, Self
from stheno.field import squeeze, mul, add, SumElement, ProductElement, \
    ScaledElement, OneElement, ZeroElement, WrappedElement, JoinElement, \
    Formatter, priority

from .field import Element, new, get_field, broadcast

__all__ = []

_dispatch = Dispatcher()


@_dispatch(B.Numeric)
def _shape(x):
    return B.shape(x)


@_dispatch(object)
def _shape(x):
    return np.shape(x)


@_dispatch(tuple, tuple)
def tuple_equal(x, y):
    """Check tuples for equality.

    Args:
        x (tuple): First tuple.
        y (tuple): Second tuple.

    Returns:
        bool: `x` and `y` are equal.
    """
    return len(x) == len(y) and \
           all([_shape(xi) == _shape(yi) and B.all(xi == yi)
                for xi, yi in zip(x, y)])


@_dispatch(B.Numeric)
def to_tensor(x):
    """Convert object to tensor.

    Args:
        x (object): Object to convert to tensor.

    Returns:
        tensor: `x` as a tensor.
    """
    return x


@_dispatch({tuple, list})
def to_tensor(x):
    return B.stack(*x, axis=0)


class Function(Element, Referentiable):
    """A function.

    Crucially, this is not a field, so that it can be inherited.
    """
    _dispatch = Dispatcher(in_class=Self)

    def stretch(self, *stretches):
        """Stretch the function.

        Args:
            *stretches (tensor): Per input, extent to stretch by.

        Returns:
            :class:`.function_field.Function`: Stretched function.
        """
        return stretch(self, *stretches)

    def __gt__(self, stretch):
        """Shorthand for :meth:`.function_field.Function.stretch`."""
        return self.stretch(stretch)

    def shift(self, *amounts):
        """Shift the inputs of an function by a certain amount.

        Args:
            *amounts (tensor): Per input, amount to shift by.

        Returns:
            :class:`.function_field.Function`: Shifted function.
        """
        return shift(self, *amounts)

    def select(self, *dims):
        """Select particular dimensions of the input features.

        Args:
            *dims (int, sequence, or None): Per input, dimensions to select.
                Set to `None` to select all.

        Returns:
            :class:`.function_field.Function`: Function with dimensions of the
                input features selected.
        """
        return select(self, *dims)

    def transform(self, *fs):
        """Transform the inputs of a function.

        Args:
            *fs (int or tuple): Per input, transformation. Set to `None` to
                not perform a transformation.

        Returns:
            :class:`.function_field.Function`: Function with its inputs
                transformed.
        """
        return transform(self, *fs)

    def diff(self, *derivs):
        """Differentiate a function.

        Args:
            *derivs (int): Per input, dimension of the feature which to take
                the derivatives with respect to. Set to `None` to not take a
                derivative.

        Returns:
            :class:`.function_field.Function`: Derivative of the Function.
        """
        return differentiate(self, *derivs)


# Register the field.
@get_field.extend(Function)
def get_field(a): return Function


class OneFunction(Function, OneElement):
    """The constant function `1`."""


class ZeroFunction(Function, ZeroElement):
    """The constant function `0`."""


class WrappedFunction(Function, WrappedElement):
    """A wrapped function."""


class JoinFunction(Function, JoinElement):
    """Two wrapped functions."""


class SumFunction(Function, SumElement):
    """A sum of two functions."""


class ScaledFunction(Function, ScaledElement):
    """A scaled function."""


class ProductFunction(Function, ProductElement):
    """A product of two functions."""


class StretchedFunction(WrappedFunction, Referentiable):
    """Stretched function.

    Args:
        e (:class:`.function_field.Function`): Function to stretch.
        *stretches (tensor): Extent of stretches.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, *stretches):
        WrappedFunction.__init__(self, e)
        self.stretches = tuple(to_tensor(x) for x in stretches)

    @_dispatch(object, Formatter)
    def display(self, e, formatter):
        stretches = tuple(formatter(s) for s in self.stretches)
        return '{} > {}'.format(e, squeeze(stretches))

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and \
               tuple_equal(self.stretches, other.stretches)


class ShiftedFunction(WrappedFunction, Referentiable):
    """Shifted function.

    Args:
        e (:class:`.function_field.Function`): Function to shift.
        *shifts (tensor): Shift amounts.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, *shifts):
        WrappedFunction.__init__(self, e)
        self.shifts = tuple(to_tensor(x) for x in shifts)

    @_dispatch(object, Formatter)
    def display(self, e, formatter):
        shifts = tuple(formatter(s) for s in self.shifts)
        return '{} shift {}'.format(e, squeeze(shifts))

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and \
               tuple_equal(self.shifts, other.shifts)


@_dispatch({tuple, list})
def _to_list(x):
    return list(x)


@_dispatch(object)
def _to_list(x):
    if B.rank(x) == 0:
        return [x]
    elif B.rank(x) == 1:
        return x
    else:
        raise ValueError('Could not convert "{}" to a list.')


class SelectedFunction(WrappedFunction, Referentiable):
    """Select particular dimensions of the input features.

    Args:
        e (:class:`.function_field.Function`): Function to wrap.
        *dims (tensor): Dimensions to select.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, *dims):
        WrappedFunction.__init__(self, e)
        self.dims = tuple(None if x is None else _to_list(x) for x in dims)

    @_dispatch(object, Formatter)
    def display(self, e, formatter):
        return '{} : {}'.format(e, squeeze(tuple(self.dims)))

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and \
               tuple_equal(self.dims, other.dims)


class InputTransformedFunction(WrappedFunction, Referentiable):
    """Transform inputs of a function.

    Args:
        e (:class:`.function_field.Function`): Function to wrap.
        *fs (tensor): Per input, the transformation. Set to `None` to not
            do a transformation.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, *fs):
        WrappedFunction.__init__(self, e)
        self.fs = fs

    @_dispatch(object, Formatter)
    def display(self, e, formatter):
        # Safely get a function's name.
        def name(f):
            return 'None' if f is None else f.__name__

        if len(self.fs) == 1:
            fs = name(self.fs[0])
        else:
            fs = '({})'.format(', '.join(name(f) for f in self.fs))
        return '{} transform {}'.format(e, fs)

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and \
               tuple_equal(self.fs, other.fs)


class DerivativeFunction(WrappedFunction, Referentiable):
    """Compute the derivative of a function.

    Args:
        e (:class:`.field_function.Element`): Function to compute the
            derivative of.
        *derivs (tensor): Per input, the index of the dimension which to
            take the derivative of. Set to `None` to not take a derivative.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, *derivs):
        WrappedFunction.__init__(self, e)
        self.derivs = derivs

    @_dispatch(object, Formatter)
    def display(self, e, formatter):
        if len(self.derivs) == 1:
            derivs = '({})'.format(self.derivs[0])
        else:
            derivs = self.derivs
        return 'd{} {}'.format(derivs, e)

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and \
               tuple_equal(self.derivs, other.derivs)


class TensorProductFunction(Function, Referentiable):
    """An element built from a product of functions for each input.

    Args:
        *fs (function): Per input, a function.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, *fs):
        self.fs = fs

    @_dispatch(Formatter)
    def display(self, formatter):
        if len(self.fs) == 1:
            return self.fs[0].__name__
        else:
            return '({})'.format(' x '.join(f.__name__ for f in self.fs))

    @_dispatch(Self)
    def __eq__(self, other):
        return tuple_equal(self.fs, other.fs)


@_dispatch(object, [object])
def stretch(a, *stretches):
    """Stretch a function.

    Args:
        a (:class:`.function_field.Function`): Function to stretch.
        *stretches (tensor): Per input, extent of stretches.

    Returns:
        :class:`.function_field.Function`: Stretched function.
    """
    raise NotImplementedError('Stretching not implemented for "{}".'
                              ''.format(type(a).__name__))


@_dispatch(object, [object])
def shift(a, *shifts):
    """Shift a function.

    Args:
        a (:class:`.function_field.Function`): Function to shift.
        *shifts (tensor): Per input, amount of shift.

    Returns:
        :class:`.function_field.Function`: Shifted element.
    """
    raise NotImplementedError('Shifting not implemented for "{}".'
                              ''.format(type(a).__name__))


@_dispatch(object, [object])
def select(a, *dims):
    """Select dimensions from the inputs.

    Args:
        a (:class:`.function_field.Function`): Function  to wrap.
        *dims (int): Per input, dimensions to select. Set to `None` to select
            all.

    Returns:
        :class:`.function_field.Function`: Function with particular dimensions
            from the inputs selected.
    """
    raise NotImplementedError('Selection not implemented for "{}".'
                              ''.format(type(a).__name__))


@_dispatch(object, [object])
def transform(a, *fs):
    """Transform the inputs of a function.

    Args:
        a (:class:`.function_field.Function`): Function to wrap.
        *fs (int): Per input, the transform. Set to `None` to not perform a
            transform.

    Returns:
        :class:`.function_field.Function`: Function with its inputs
            transformed.
    """
    raise NotImplementedError('Input transforms not implemented for "{}".'
                              ''.format(type(a).__name__))


@_dispatch(object, [object])
def differentiate(a, *derivs):
    """Differentiate a function.

    Args:
        a (:class:`.function_field.Function`): Function to differentiate.
        *derivs (int): Per input, dimension of the feature which to take
            the derivatives with respect to. Set to `None` to not take a
            derivative.

    Returns:
        :class:`.function_field.Function`: Derivative of the function.
    """
    raise NotImplementedError('Differentiation not implemented for "{}".'
                              ''.format(type(a).__name__))


# Handle conversion of Python functions.

@mul.extend(Element, PythonFunction, precedence=priority)
def mul(a, b): return mul(a, new(a, TensorProductFunction)(b))


@mul.extend(PythonFunction, Element, precedence=priority)
def mul(a, b): return mul(new(b, TensorProductFunction)(a), b)


@add.extend(Element, PythonFunction, precedence=priority)
def add(a, b): return add(a, new(a, TensorProductFunction)(b))


@add.extend(PythonFunction, Element, precedence=priority)
def add(a, b): return add(new(b, TensorProductFunction)(a), b)


# Stretch:

@_dispatch(Element, [object])
def stretch(a, *stretches): return new(a, StretchedFunction)(a, *stretches)


@_dispatch({ZeroElement, OneElement}, [object])
def stretch(a, *stretches): return a


@_dispatch(StretchedFunction, [object])
def stretch(a, *stretches):
    return stretch(a[0], *broadcast(operator.mul, a.stretches, stretches))


# Shifting:

@_dispatch(Element, [object])
def shift(a, *shifts): return new(a, ShiftedFunction)(a, *shifts)


@_dispatch({ZeroElement, OneElement}, [object])
def shift(a, *shifts): return a


@_dispatch(ShiftedFunction, [object])
def shift(a, *shifts):
    return shift(a[0], *broadcast(operator.add, a.shifts, shifts))


# Selection:

@_dispatch(Element, [object])
def select(a, *dims): return new(a, SelectedFunction)(a, *dims)


@_dispatch({ZeroElement, OneElement}, [object])
def select(a, *dims): return a


# Input transforms:

@_dispatch(Element, [object])
def transform(a, *fs): return new(a, InputTransformedFunction)(a, *fs)


@_dispatch({ZeroElement, OneElement}, [object])
def transform(a, *fs): return a


# Differentiation:

@_dispatch(Element, [object])
def differentiate(a, *derivs): return new(a, DerivativeFunction)(a, *derivs)


@_dispatch(ZeroElement, [object])
def differentiate(a, *derivs): return a


@_dispatch(OneElement, [object])
def differentiate(a, *derivs): return new(a, ZeroElement)()
