# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import operator
from types import FunctionType as PythonFunction
from plum import Dispatcher, Referentiable, Self

from stheno.field import squeeze, mul, add, SumElement, ProductElement, \
    ScaledElement, OneElement, ZeroElement, WrappedElement, PrimitiveElement, \
    JoinElement, Element
from .field import dispatch_field, Element, new, get_field, broadcast

__all__ = []


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

    def shift(self, *amounts):
        """Shift the inputs of an function by a certain amount.

        Args:
            *amounts (tensor): Per input, amount to shift by.

        Returns:
            :class:`.function_field.Function`: Shifted function.
        """
        return shift(self, *amounts)

    @_dispatch([{tuple, list, type(None)}])
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

    @_dispatch([object])
    def select(self, *dims):
        return select(self, dims)

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
@dispatch_field(Function)
def get_field(a): return Function


class PrimitiveFunction(PrimitiveElement):
    """A primitive function."""


class OneFunction(OneElement):
    """The constant function `1`."""


class ZeroFunction(ZeroElement):
    """The constant function `0`."""


class WrappedFunction(WrappedElement):
    """A wrapped function."""


class JoinFunction(JoinElement):
    """Two wrapped functions."""


class SumFunction(SumElement):
    """A sum of two functions."""


class ScaledFunction(ScaledElement):
    """A scaled function."""


class ProductFunction(ProductElement):
    """A product of two functions."""


class StretchedFunction(WrappedFunction):
    """Stretched function.

    Args:
        e (:class:`.function_field.Function`): Function to stretch.
        *stretches (tensor): Extent of stretches.
    """

    def __init__(self, e, *stretches):
        WrappedFunction.__init__(self, e)
        self.stretches = stretches

    def display(self, e):
        return '{} > {}'.format(e, squeeze(self.stretches))


class ShiftedFunction(WrappedFunction):
    """Shifted function.

    Args:
        e (:class:`.function_field.Function`): Function to shift.
        *shifts (tensor): Shift amounts.
    """

    def __init__(self, e, *shifts):
        WrappedFunction.__init__(self, e)
        self.shifts = shifts

    def display(self, e):
        return '{} shift {}'.format(e, squeeze(self.shifts))


class SelectedFunction(WrappedFunction):
    """Select particular dimensions of the input features.

    Args:
        e (:class:`.function_field.Function`): Function to wrap.
        *dims (tensor): Dimensions to select.
    """

    def __init__(self, e, *dims):
        WrappedFunction.__init__(self, e)
        self.dims = dims

    def display(self, e):
        return '{} : {}'.format(e, squeeze(tuple(list(ds) for ds in self.dims)))


class InputTransformedFunction(WrappedFunction):
    """Transform inputs of a function.

    Args:
        e (:class:`.function_field.Function`): Function to wrap.
        *fs (tensor): Per input, the transformation. Set to `None` to not
            do a transformation.
    """

    def __init__(self, e, *fs):
        WrappedFunction.__init__(self, e)
        self.fs = fs

    def display(self, e):
        # Safely get a function's name.
        def name(f):
            return 'None' if f is None else f.__name__

        if len(self.fs) == 1:
            fs = name(self.fs[0])
        else:
            fs = '({})'.format(', '.join(name(f) for f in self.fs))
        return '{} transform {}'.format(e, fs)


class DerivativeFunction(WrappedFunction):
    """Compute the derivative of a function.

    Args:
        e (:class:`.field_function.Element`): Function to compute the
            derivative of.
        *derivs (tensor): Per input, the index of the dimension which to
            take the derivative of. Set to `None` to not take a derivative.
    """

    def __init__(self, e, *derivs):
        WrappedFunction.__init__(self, e)
        self.derivs = derivs

    def display(self, e):
        if len(self.derivs) == 1:
            derivs = '({})'.format(self.derivs[0])
        else:
            derivs = self.derivs
        return 'd{} {}'.format(derivs, e)


class TensorProductFunction(Function):
    """An element built from a product of functions for each input.

    Args:
        *fs (function): Per input, a function.
    """

    def __init__(self, *fs):
        self.fs = fs

    def __str__(self):
        if len(self.fs) == 1:
            return self.fs[0].__name__
        else:
            return '({})'.format(' x '.join(f.__name__ for f in self.fs))


@dispatch_field(object, [object])
def stretch(a, *stretches):
    """Stretch a function.

    Args:
        a (:class:`.function_field.Function`): Function to stretch.
        *stretches (tensor): Per input, extent of stretches.

    Returns:
        :class:`.function_field.Function`: Stretched function.
    """
    raise NotImplementedError('Stretching not implemented for {}.'
                              ''.format(type(a).__name__))


@dispatch_field(object, [object])
def shift(a, *shifts):
    """Shift a function.

    Args:
        a (:class:`.function_field.Function`): Function to shift.
        *shifts (tensor): Per input, amount of shift.

    Returns:
        :class:`.function_field.Function`: Shifted element.
    """
    raise NotImplementedError('Shifting not implemented for {}.'
                              ''.format(type(a).__name__))


@dispatch_field(object, [object])
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
    raise NotImplementedError('Selection not implemented for {}.'
                              ''.format(type(a).__name__))


@dispatch_field(object, [object])
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
    raise NotImplementedError('Input transforms not implemented for {}.'
                              ''.format(type(a).__name__))


@dispatch_field(object, [object])
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
    raise NotImplementedError('Differentiation not implemented for {}.'
                              ''.format(type(a).__name__))


# Handle conversion of Python functions.

@dispatch_field(Element, PythonFunction, precedence=1)
def mul(a, b): return mul(a, new(a, TensorProductFunction)(b))


@dispatch_field(PythonFunction, Element, precedence=1)
def mul(a, b): return mul(new(b, TensorProductFunction)(a), b)


@dispatch_field(Element, PythonFunction, precedence=1)
def add(a, b): return add(a, new(a, TensorProductFunction)(b))


@dispatch_field(PythonFunction, Element, precedence=1)
def add(a, b): return add(new(b, TensorProductFunction)(a), b)


# Stretch:

@dispatch_field(Element, [object])
def stretch(a, *stretches): return new(a, StretchedFunction)(a, *stretches)


@dispatch_field({ZeroElement, OneElement}, [object])
def stretch(a, *stretches): return a


@dispatch_field(StretchedFunction, [object])
def stretch(a, *stretches):
    return stretch(a[0], *broadcast(operator.mul, a.stretches, stretches))


# Shifting:

@dispatch_field(Element, [object])
def shift(a, *shifts): return new(a, ShiftedFunction)(a, *shifts)


@dispatch_field({ZeroElement, OneElement}, [object])
def shift(a, *shifts): return a


@dispatch_field(ShiftedFunction, [object])
def shift(a, *shifts):
    return shift(a[0], *broadcast(operator.add, a.shifts, shifts))


# Selection:

@dispatch_field(Element, [object])
def select(a, *dims): return new(a, SelectedFunction)(a, *dims)


@dispatch_field({ZeroElement, OneElement}, [object])
def select(a, *dims): return a


# Input transforms:

@dispatch_field(Element, [object])
def transform(a, *fs): return new(a, InputTransformedFunction)(a, *fs)


@dispatch_field({ZeroElement, OneElement}, [object])
def transform(a, *fs): return a


# Differentiation:

@dispatch_field(Element, [object])
def differentiate(a, *derivs): return new(a, DerivativeFunction)(a, *derivs)


@dispatch_field(ZeroElement, [object])
def differentiate(a, *derivs): return a


@dispatch_field(OneElement, [object])
def differentiate(a, *derivs): return new(a, ZeroElement)()
