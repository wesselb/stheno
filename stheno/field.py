# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import operator
from types import FunctionType as Function

from plum import Dispatcher, Referentiable, Self, NotFoundLookupError

__all__ = []

_dispatch = Dispatcher()


def apply_optional_arg(f, arg1, arg2):
    """If `f` takes in two or more arguments, run `f(arg1, arg2)`; otherwise,
    run `f(arg1)`.

    Args:
        f (function): Function to run.
        arg1 (object): First argument for `f`.
        arg2 (object): Optional argument for `f`.

    Returns:
        object: Result of `f(arg1, arg2)` or `f(arg1)`.
    """
    try:
        return f(arg1, arg2)
    except TypeError:
        return f(arg1)
    except NotFoundLookupError:
        return f(arg1)


def squeeze(xs):
    """Squeeze an sequence if it only contains a single element.

    Args:
        xs (sequence): Sequence to squeeze.

    Returns:
        object: `xs[0]` if `xs` consists of a single element and `xs` otherwise.
    """
    return xs[0] if len(xs) == 1 else xs


def broadcast(op, xs, ys):
    """Perform a binary operation `op` on elements of `xs` and `ys`. If `xs` or
    `ys` has length 1, then it is repeated sufficiently many times to match the
    length of the other.

    Args:
        op (function): Binary operation.
        xs (sequence): First sequence.
        ys (sequence): Second sequence.

    Returns:
        tuple: Result of applying `op` to every element of `zip(xs, ys)` after
        broadcasting appropriately.
    """
    if len(xs) == 1 and len(ys) > 1:
        # Broadcast `xs`.
        xs = xs * len(ys)
    elif len(ys) == 1 and len(xs) > 1:
        # Broadcast `ys.
        ys = ys * len(xs)

    # Check that `xs` and `ys` are compatible now.
    if len(xs) != len(ys):
        raise ValueError('Inputs "{}" and "{}" could not be broadcasted.'
                         ''.format(xs, ys))
    # Perform operation.
    return tuple(op(x, y) for x, y in zip(xs, ys))


class Type(Referentiable):
    """A field over functions.

    Functions are also referred to as elements of the field. Elements can be
    added and multiplied.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __eq__(self, other):
        return equal(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __neg__(self):
        return mul(-1, self)

    def __sub__(self, other):
        return add(self, -other)

    def __rsub__(self, other):
        return add(other, -self)

    def stretch(self, *stretches):
        """Stretch the element.

        Args:
            *stretches (tensor): Per input, extent to stretch by.

        Returns:
            :class:`.field.Type`: Stretched element.
        """
        return stretch(self, *stretches)

    def shift(self, *amounts):
        """Shift the inputs of an element by a certain amount.

        Args:
            *amounts (tensor): Per input, amount to shift by.

        Returns:
            :class:`.field.Type`: Shifted element.
        """
        return shift(self, *amounts)

    @_dispatch([{tuple, list, type(None)}])
    def select(self, *dims):
        """Select particular dimensions of the input features.

        Args:
            *dims (int, sequence, or None): Per input, dimensions to select.
                Set to `None` to select all.

        Returns:
            :class:`.field.Type`: Element with dimensions of the input features
            selected.
        """
        return select(self, *dims)

    @_dispatch([object])
    def select(self, *dims):
        return select(self, dims)

    def transform(self, *fs):
        """Transform the inputs.

        Args:
            *fs (int or tuple): Per input, transformation. Set to `None` to
                not perform a transformation.

        Returns:
            :class:`.field.Type`: Element with its inputs transformed.
        """
        return transform(self, *fs)

    def diff(self, *derivs):
        """Differentiate.

        Args:
            *derivs (int): Per input, dimension of the feature which to take
                the derivatives with respect to. Set to `None` to not take a
                derivative.

        Returns:
            :class:`.field.Type`: Derivative of the element.
        """
        return differentiate(self, *derivs)

    @property
    def num_terms(self):
        """Number of terms"""
        return 1

    def term(self, i):
        """Get a specific term.

        Args:
            i (int): Index of term.

        Returns:
            :class:`.field.Type`: The referenced term.
        """
        if i == 0:
            return self
        else:
            raise IndexError('Index out of range.')

    @property
    def num_factors(self):
        """Number of factors"""
        return 1

    def factor(self, i):
        """Get a specific factor.

        Args:
            i (int): Index of factor.

        Returns:
            :class:`.field.Type`: The referenced factor.
        """
        if i == 0:
            return self
        else:
            raise IndexError('Index out of range.')

    def __repr__(self):
        return str(self)

    @property
    def __name__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__ + '()'


class PrimitiveType(Type):
    """A primitive.

    Instances of primitives should always be the same element. It therefore
    does not make sense if primitive elements have external parameters.
    """


class OneType(PrimitiveType):
    """The constant `1`."""

    def __str__(self):
        return '1'


class ZeroType(PrimitiveType):
    """The constant `0`."""

    def __str__(self):
        return '0'


class WrappedType(Type):
    """A wrapped element.

    Args:
        t (:class:`.field.Type`): Element to wrap.
    """

    def __init__(self, t):
        self.t = t

    def __getitem__(self, item):
        if item == 0:
            return self.t
        else:
            raise IndexError('Index out of range.')

    def display(self, t):
        raise NotImplementedError()

    def __str__(self):
        return pretty_print(self)


class JoinType(Type):
    """Two wrapped elements.

    Args:
        t1 (:class:`.field.Type`): First element to wrap.
        t2 (:class:`.field.Type`): Second element to wrap.
    """

    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def __getitem__(self, item):
        if item == 0:
            return self.t1
        elif item == 1:
            return self.t2
        else:
            raise IndexError('Index out of range.')

    def display(self, t1, t2):
        raise NotImplementedError()

    def __str__(self):
        return pretty_print(self)


class ScaledType(WrappedType):
    """Scaled element.

    Args:
        t (:class:`.field.Type`): Element to scale.
        scale (tensor): Scale.
    """

    def __init__(self, t, scale):
        WrappedType.__init__(self, t)
        self.scale = scale

    @property
    def num_factors(self):
        return self[0].num_factors + 1

    def display(self, t):
        return '{} * {}'.format(self.scale, t)

    def factor(self, i):
        if i >= self.num_factors:
            raise IndexError('Index out of range.')
        else:
            return self.scale if i == 0 else self[0].factor(i - 1)


class StretchedType(WrappedType):
    """Stretched element.

    Args:
        t (:class:`.field.Type`): Element to stretch.
        *stretches (tensor): Extent of stretches.
    """

    def __init__(self, t, *stretches):
        WrappedType.__init__(self, t)
        self.stretches = stretches

    def display(self, t):
        return '{} > {}'.format(t, squeeze(self.stretches))


class ShiftedType(WrappedType):
    """Shifted element.

    Args:
        t (:class:`.field.Type`): Element to shift.
        *shifts (tensor): Shift amounts.
    """

    def __init__(self, t, *shifts):
        WrappedType.__init__(self, t)
        self.shifts = shifts

    def display(self, t):
        return '{} shift {}'.format(t, squeeze(self.shifts))


class SelectedType(WrappedType):
    """Select particular dimensions of the input features.

    Args:
        t (:class:`.field.Type`): Element to wrap.
        *dims (tensor): Dimensions to select.
    """

    def __init__(self, t, *dims):
        WrappedType.__init__(self, t)
        self.dims = dims

    def display(self, t):
        return '{} : {}'.format(t, squeeze(tuple(list(ds) for ds in self.dims)))


class InputTransformedType(WrappedType):
    """Transform the inputs for a particular element.

    Args:
        t (:class:`.field.Type`): Element to wrap.
        *fs (tensor): Per input, the transformation. Set to `None` to not
            do a transformation.
    """

    def __init__(self, t, *fs):
        WrappedType.__init__(self, t)
        self.fs = fs

    def display(self, t):
        # Safely get a function's name.
        def name(f):
            return 'None' if f is None else f.__name__

        if len(self.fs) == 1:
            fs = name(self.fs[0])
        else:
            fs = '({})'.format(', '.join(name(f) for f in self.fs))
        return '{} transform {}'.format(t, fs)


class DerivativeType(WrappedType):
    """Compute the derivative with respect to the inputs of an element.

    Args:
        t (:class:`.field.Type`): Element to compute
            the derivative of.
        *derivs (tensor): Per input, the index of the dimension which to
            take the derivative of. Set to `None` to not take a derivative.
    """

    def __init__(self, t, *derivs):
        WrappedType.__init__(self, t)
        self.derivs = derivs

    def display(self, t):
        if len(self.derivs) == 1:
            derivs = '({})'.format(self.derivs[0])
        else:
            derivs = self.derivs
        return 'd{} {}'.format(derivs, t)


class ProductType(JoinType):
    """Product of elements."""

    @property
    def num_factors(self):
        return self[0].num_factors + self[1].num_factors

    def factor(self, i):
        if i >= self.num_factors:
            raise IndexError('Index out of range.')
        if i < self[0].num_factors:
            return self[0].factor(i)
        else:
            return self[1].factor(i - self[0].num_factors)

    def display(self, t1, t2):
        return '{} * {}'.format(t1, t2)


class SumType(JoinType):
    """Sum of elements."""

    @property
    def num_terms(self):
        return self[0].num_terms + self[1].num_terms

    def term(self, i):
        if i >= self.num_terms:
            raise IndexError('Index out of range.')
        if i < self[0].num_terms:
            return self[0].term(i)
        else:
            return self[1].term(i - self[0].num_terms)

    def display(self, t1, t2):
        return '{} + {}'.format(t1, t2)


class FunctionType(Type):
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


@_dispatch(object, object)
def mul(a, b):
    """Multiply two elements.

    Args:
        a (:class:`.field.Type`): First element in product.
        b (:class:`.field.Type`): Second element in product.

    Returns:
        :class:`.field.Type`: Product of the elements.
    """
    raise NotImplementedError('Multiplication not implemented for {} and {}.'
                              ''.format(type(a).__name__, type(b).__name__))


@_dispatch(object, object)
def add(a, b):
    """Add two elements.

    Args:
        a (:class:`.field.Type`): First element in summation.
        b (:class:`.field.Type`): Second element in summation.

    Returns:
        :class:`.field.Type`: Sum of the elements.
    """
    raise NotImplementedError('Addition not implemented for {} and {}.'
                              ''.format(type(a).__name__, type(b).__name__))


@_dispatch(object, [object])
def stretch(a, *stretches):
    """Stretch an element.

    Args:
        a (:class:`.field.Type`): Element to stretch.
        *stretches (tensor): Per input, extent of stretches.

    Returns:
        :class:`.field.Type`: Stretched element.
    """
    raise NotImplementedError('Stretching not implemented for {}.'
                              ''.format(type(a).__name__))


@_dispatch(object, [object])
def shift(a, *shifts):
    """Shift an element.

    Args:
        a (:class:`.field.Type`): Element to shift.
        *shifts (tensor): Per input, amount of shift.

    Returns:
        Shifted element.
    """
    raise NotImplementedError('Shifting not implemented for {}.'
                              ''.format(type(a).__name__))


@_dispatch(object, [object])
def select(a, *dims):
    """Select dimensions from the inputs.

    Args:
        a (:class:`.field.Type`): Element to wrap.
        *dims (int): Per input, dimensions to select. Set to `None` to select
            all.

    Returns:
        Element with particular dimensions from the inputs selected.
    """
    raise NotImplementedError('Selection not implemented for {}.'
                              ''.format(type(a).__name__))


@_dispatch(object, [object])
def transform(a, *fs):
    """Transform inputs.

    Args:
        a (:class:`.field.Type`): Element to wrap.
        *fs (int): Per input, the transform. Set to `None` to not perform a
            transform.

    Returns:
        :class:`.field.Type`: Element with its inputs transformed.
    """
    raise NotImplementedError('Input transforms not implemented for {}.'
                              ''.format(type(a).__name__))


@_dispatch(object, [object])
def differentiate(a, *derivs):
    """Differentiate.

    Args:
        a (:class:`.field.Type`): Element to differentiate.
        *derivs (int): Per input, dimension of the feature which to take
            the derivatives with respect to. Set to `None` to not take a
            derivative.

    Returns:
        :class:`.field.Type`: Derivative of the element.
    """
    raise NotImplementedError('Differentiation not implemented for {}.'
                              ''.format(type(a).__name__))


field_types = set(Type.__subclasses__())
field_cache = {}


def get_field(a):
    """Get the field type of an element.

    Args:
        a (:class:`.field.Type`): Element to get field of.

    Returns:
        type: Field type of `a`.
    """
    try:
        return field_cache[type(a)]
    except KeyError:
        # Figure out which fields are defined.
        fields = set(Type.__subclasses__()) - field_types

        # Loop through the fields, and see to which one `a` corresponds.
        candidates = []
        for t in fields:
            if isinstance(a, t):
                candidates.append(t)

        # There should only be a single candidate.
        if len(candidates) != 1:
            raise RuntimeError('Could not determine field type of {}.'
                               ''.format(type(a).__name__))
        field_cache[type(a)] = candidates[0]
        return field_cache[type(a)]


new_cache = {}


def new(a, t):
    """Create a new specialised type.

    Args:
        a (:class:`.field.Type`): Element to create new type for.
        t (type): Type to create.

    Returns:
        type: Specialisation of `t` appropriate for `a`.
    """
    try:
        return new_cache[type(a), t]
    except KeyError:
        field = get_field(a)
        candidates = list(set(field.__subclasses__()) & set(t.__subclasses__()))

        # There should only be a single candidate.
        if len(candidates) != 1:
            raise RuntimeError('Could not determine {} for field {}.'
                               ''.format(t.__name__, field.__name__))

        new_cache[type(a), t] = candidates[0]
        return new_cache[type(a), t]


# Pretty printing with minimal parentheses.

@_dispatch(Type)
def pretty_print(el):
    """Pretty print an element with a minimal number of parentheses.

    Args:
        el (:class:`.field.Type`): Element to print.

    Returns:
        str: `el` converted to string prettily.
    """
    return str(el)


@_dispatch(WrappedType)
def pretty_print(el):
    return el.display(pretty_print(el[0], el))


@_dispatch(JoinType)
def pretty_print(el):
    return el.display(pretty_print(el[0], el), pretty_print(el[1], el))


@_dispatch(object, object)
def pretty_print(el, parent):
    if need_parens(el, parent):
        return '(' + pretty_print(el) + ')'
    else:
        return pretty_print(el)


@_dispatch(Type, SumType)
def need_parens(el, parent):
    """Check whether `el` needs parentheses when printed in `parent`.

    Args:
        el (:class:`.field.Type`): Element to print.
        parent (:class:`.field.Type`): Parent of element to print.

    Returns:
        bool: Boolean whether `el` needs parentheses.
    """
    return False


@_dispatch(Type, ProductType)
def need_parens(el, parent): return False


@_dispatch({SumType, WrappedType}, ProductType)
def need_parens(el, parent): return True


@_dispatch(ScaledType, ProductType)
def need_parens(el, parent): return False


@_dispatch(Type, WrappedType)
def need_parens(el, parent): return False


@_dispatch({WrappedType, JoinType}, WrappedType)
def need_parens(el, parent): return True


@_dispatch({ProductType, ScaledType}, ScaledType)
def need_parens(el, parent): return False


# Generic multiplication.

@_dispatch(object, Type)
def mul(a, b): return mul(b, a)


@_dispatch(Type, object)
def mul(a, b):
    if b == 0:
        return new(a, ZeroType)()
    elif b == 1:
        return a
    else:
        return new(a, ScaledType)(a, b)


@_dispatch(Type, Function)
def mul(a, b): return mul(a, new(a, FunctionType)(b))


@_dispatch(Function, Type)
def mul(a, b): return mul(new(b, FunctionType)(a), b)


@_dispatch(Type, Type)
def mul(a, b): return new(a, ProductType)(a, b)


# Generic addition.

@_dispatch(object, Type)
def add(a, b):
    if a == 0:
        return b
    else:
        return new(b, SumType)(mul(a, new(b, OneType)()), b)


@_dispatch(Function, Type)
def add(a, b): return add(new(b, FunctionType)(a), b)


@_dispatch(Type, object)
def add(a, b):
    if b == 0:
        return a
    else:
        return new(a, SumType)(a, mul(b, new(a, OneType)()))


@_dispatch(Type, Function)
def add(a, b): return add(a, new(a, FunctionType)(b))


@_dispatch(Type, Type)
def add(a, b):
    if a == b:
        return mul(2, a)
    else:
        return new(a, SumType)(a, b)


# Cancel redundant zeros and ones.

@_dispatch.multi((ZeroType, object),
                (ZeroType, Function),
                (ZeroType, Type),
                (ZeroType, ScaledType),
                (ZeroType, ZeroType),

                (Type, OneType),
                (ScaledType, OneType),
                (OneType, OneType))
def mul(a, b): return a


@_dispatch.multi((object, ZeroType),
                (Function, ZeroType),
                (Type, ZeroType),
                (ScaledType, ZeroType),

                (OneType, Type),
                (OneType, ScaledType))
def mul(a, b): return b


@_dispatch.multi((ZeroType, Type),
                (ZeroType, ScaledType),
                (ZeroType, ZeroType))
def add(a, b): return b


@_dispatch.multi((Type, ZeroType),
                (ScaledType, ZeroType))
def add(a, b): return a


@_dispatch(object, ZeroType)
def add(a, b): return add(b, a)


@_dispatch(Function, ZeroType)
def add(a, b): return new(b, FunctionType)(a)


@_dispatch(ZeroType, Function)
def add(b, a): return new(b, FunctionType)(a)


@_dispatch(ZeroType, object)
def add(a, b):
    if b == 0:
        return a
    elif b == 1:
        return new(a, OneType)()
    else:
        return new(a, ScaledType)(new(a, OneType)(), b)


# Group factors and terms if possible.

@_dispatch(object, ScaledType)
def mul(a, b): return mul(b, a)


@_dispatch(ScaledType, object)
def mul(a, b): return mul(a.scale * b, a[0])


@_dispatch(ScaledType, Function)
def mul(a, b): return mul(a, new(a, FunctionType)(b))


@_dispatch(Function, ScaledType)
def mul(a, b): return mul(new(b, FunctionType)(a), b)


@_dispatch(ScaledType, Type)
def mul(a, b): return mul(a.scale, a[0] * b)


@_dispatch(Type, ScaledType)
def mul(a, b): return mul(b.scale, a * b[0])


@_dispatch(ScaledType, ScaledType)
def mul(a, b):
    if a[0] == b[0]:
        return new(a, ScaledType)(a[0], a.scale * b.scale)
    else:
        scaled = new(a, ScaledType)(a[0], a.scale * b.scale)
        return new(a, ProductType)(scaled, b[0])


@_dispatch(ScaledType, Type)
def add(a, b):
    if a[0] == b:
        return mul(a.scale + 1, b)
    else:
        return new(a, SumType)(a, b)


@_dispatch(Type, ScaledType)
def add(a, b):
    if a == b[0]:
        return mul(b.scale + 1, a)
    else:
        return new(a, SumType)(a, b)


@_dispatch(ScaledType, ScaledType)
def add(a, b):
    if a[0] == b[0]:
        return mul(a.scale + b.scale, a[0])
    else:
        return new(a, SumType)(a, b)


# Stretch:

@_dispatch(Type, [object])
def stretch(a, *stretches): return new(a, StretchedType)(a, *stretches)


@_dispatch({ZeroType, OneType}, [object])
def stretch(a, *stretches): return a


@_dispatch(StretchedType, [object])
def stretch(a, *stretches):
    return stretch(a[0], *broadcast(operator.mul, a.stretches, stretches))


# Shifting:

@_dispatch(Type, [object])
def shift(a, *shifts): return new(a, ShiftedType)(a, *shifts)


@_dispatch({ZeroType, OneType}, [object])
def shift(a, *shifts): return a


@_dispatch(ShiftedType, [object])
def shift(a, *shifts):
    return shift(a[0], *broadcast(operator.add, a.shifts, shifts))


# Selection:

@_dispatch(Type, [object])
def select(a, *dims): return new(a, SelectedType)(a, *dims)


@_dispatch({ZeroType, OneType}, [object])
def select(a, *dims): return a


# Input transforms:

@_dispatch(Type, [object])
def transform(a, *fs): return new(a, InputTransformedType)(a, *fs)


@_dispatch({ZeroType, OneType}, [object])
def transform(a, *fs): return a


# Differentiation:

@_dispatch(Type, [object])
def differentiate(a, *derivs): return new(a, DerivativeType)(a, *derivs)


@_dispatch(ZeroType, [object])
def differentiate(a, *derivs): return a


@_dispatch(OneType, [object])
def differentiate(a, *derivs): return new(a, ZeroType)()


# Equality:

@_dispatch(object, object)
def equal(a, b): return False


@_dispatch(PrimitiveType, PrimitiveType)
def equal(a, b): return type(a) == type(b)


@_dispatch.multi((SumType, SumType),
                (ProductType, ProductType))
def equal(a, b): return (equal(a[0], b[0]) and equal(a[1], b[1])) or \
                        (equal(a[0], b[1]) and equal(a[1], b[0]))
