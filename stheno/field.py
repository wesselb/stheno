# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from plum import Dispatcher

__all__ = ['dispatch', 'Type', 'PrimitiveType', 'OneType', 'ZeroType',
           'WrappedType', 'ScaledType', 'StretchedType', 'ProductType',
           'SumType', 'mul', 'add', 'stretch', 'equal']

dispatch = Dispatcher()


class Type(object):
    """A field."""

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

    def stretch(self, extent):
        """Stretch the element.

        Args:
            extent (tensor): Extent to stretch by.
        """
        return stretch(self, extent)


class PrimitiveType(Type):
    """A primitive."""


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
        t (instance of :class:`.field.Type`): Element to wrap.
    """

    def __init__(self, t):
        self.t = t

    def __getitem__(self, item):
        if item == 0:
            return self.t
        raise RuntimeError('Can only get item 0.')


class ScaledType(WrappedType):
    """Scaled element.

    Args:
        t (instance of :class:`.field.Type`): Element to scale.
        scale (tensor): Scale.
    """

    def __init__(self, t, scale):
        WrappedType.__init__(self, t)
        self.scale = scale

    def __str__(self):
        return '({} * {})'.format(self.scale, self[0])


class StretchedType(WrappedType):
    """Stretched element.

    Args:
        t (instance of :class:`.field.Type`): Element to stretch.
        extent (tensor): Extend of stretch.
    """

    def __init__(self, t, extent):
        WrappedType.__init__(self, t)
        self.extent = extent

    def __str__(self):
        return '({} > {})'.format(self[0], self.extent)


class ProductType(Type):
    """Product of elements.

    Args:
        t1 (instance of :class:`.field.Type`): First element in product.
        t2 (instance of :class:`.field.Type`): Second element in product.
    """

    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def __getitem__(self, item):
        if item == 0:
            return self.t1
        if item == 1:
            return self.t2
        raise RuntimeError('Can only get items 0 or 1.')

    def __str__(self):
        return '({} * {})'.format(self[0], self[1])


class SumType(Type):
    """Sum of elements.

    Args:
        t1 (instance of :class:`.field.Type`): First element in sum.
        t2 (instance of :class:`.field.Type`): Second element in sum.
    """

    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def __getitem__(self, item):
        if item == 0:
            return self.t1
        if item == 1:
            return self.t2
        raise RuntimeError('Can only get items 0 or 1.')

    def __str__(self):
        return '({} + {})'.format(self[0], self[1])


@dispatch(object, object)
def mul(a, b):
    """Multiply two elements.

    a (instance of :class:`.field.Type`): First element in product.
    b (instance of :class:`.field.Type`): Second element in product.
    """
    raise NotImplementedError('Multiplication not implemented for {} and {}.'
                              ''.format(type(a).__name__, type(b).__name__))


@dispatch(object, object)
def add(a, b):
    """Sum two elements.

    a (instance of :class:`.field.Type`): First element in summation.
    b (instance of :class:`.field.Type`): Second element in summation.
    """
    raise NotImplementedError('Addition not implemented for {} and {}.'
                              ''.format(type(a).__name__, type(b).__name__))


@dispatch(object, object)
def stretch(a, b):
    """Stretch an element.

    a (instance of :class:`.field.Type`): Element to stretch.
    b (tensor): Extent of stretch.
    """
    raise NotImplementedError('Stretching not implemented for {}.'
                              ''.format(type(a).__name__))


def get_field(a):
    """Get the field type of an element.

    Args:
        a (instance of :class:`.field.Type`): Element to get field of.
    """
    # Figure out which fields are defined.
    fields = set(Type.__subclasses__()) - \
             {PrimitiveType, ZeroType, OneType, ScaledType, WrappedType,
              ProductType, SumType, StretchedType}

    # Loop through the fields, and see to which one `a` corresponds.
    for t in fields:
        if isinstance(a, t):
            return t
    raise RuntimeError('Could not determine field type of {}.'
                       ''.format(type(a).__name__))


def new(a, t):
    """Create a new specialised type.

    Args:
        a (instance of :class:`.field.Type`): Element to create new type for.
        t (type): Type to create.
    """
    field = get_field(a)
    candidates = list(set(field.__subclasses__()) & set(t.__subclasses__()))
    if len(candidates) != 1:
        raise RuntimeError('Could not determine {} for field {}.'
                           ''.format(t.__name__, field.__name__))
    else:
        return candidates[0]


# Generic multiplication.

@dispatch(object, Type)
def mul(a, b): return mul(b, a)


@dispatch(Type, object)
def mul(a, b):
    if b == 0:
        return new(a, ZeroType)()
    elif b == 1:
        return a
    else:
        return new(a, ScaledType)(a, b)


@dispatch(Type, Type)
def mul(a, b): return new(a, ProductType)(a, b)


# Generic addition.

@dispatch(object, Type)
def add(a, b): return add(b, a)


@dispatch(Type, object)
def add(a, b):
    if b == 0:
        return a
    else:
        return new(a, SumType)(a, mul(b, new(a, OneType)()))


@dispatch(Type, Type)
def add(a, b):
    if a == b:
        return mul(2, a)
    else:
        return new(a, SumType)(a, b)


# Cancel redundant zeros and ones.

@dispatch.multi((ZeroType, object),
                (ZeroType, Type),
                (ZeroType, ScaledType),
                (ZeroType, ZeroType),

                (Type, OneType),
                (ScaledType, OneType),
                (OneType, OneType))
def mul(a, b): return a


@dispatch.multi((object, ZeroType),
                (Type, ZeroType),
                (ScaledType, ZeroType),

                (OneType, Type),
                (OneType, ScaledType))
def mul(a, b): return b


@dispatch.multi((ZeroType, object),
                (ZeroType, Type),
                (ZeroType, ScaledType),
                (ZeroType, ZeroType))
def add(a, b): return b


@dispatch.multi((object, ZeroType),
                (Type, ZeroType),
                (ScaledType, ZeroType))
def add(a, b): return a


# Group factors and terms if possible.

@dispatch.multi((object, ScaledType),
                (Type, ScaledType))
def mul(a, b): return mul(b, a)


@dispatch(ScaledType, object)
def mul(a, b): return mul(a.scale * b, a[0])


@dispatch(ScaledType, Type)
def mul(a, b): return mul(a.scale, a[0] * b)


@dispatch(ScaledType, ScaledType)
def mul(a, b):
    if a[0] == b[0]:
        return new(a, ScaledType)(a[0], a.scale * b.scale)
    else:
        return new(a, ProductType)(a, b)


@dispatch(ScaledType, Type)
def add(a, b):
    if a[0] == b:
        return mul(a.scale + 1, b)
    else:
        return new(a, SumType)(a, b)


@dispatch(Type, ScaledType)
def add(a, b):
    if a == b[0]:
        return mul(b.scale + 1, a)
    else:
        return new(a, SumType)(a, b)


@dispatch(ScaledType, ScaledType)
def add(a, b):
    if a[0] == b[0]:
        return mul(a.scale + b.scale, a[0])
    else:
        return new(a, SumType)(a, b)


# Distributive property:

@dispatch.multi((object, SumType),
                (Type, SumType))
def mul(a, b): return add(mul(a, b[0]), mul(a, b[1]))


@dispatch.multi((SumType, object),
                (SumType, Type))
def mul(a, b): return add(mul(a[0], b), mul(a[1], b))


@dispatch(SumType, SumType)
def mul(a, b): return add(add(mul(a[0], b[0]), mul(a[0], b[1])),
                          add(mul(a[1], b[0]), mul(a[1], b[1])))


# Stretch:

@dispatch(Type, object)
def stretch(a, b): return new(a, StretchedType)(a, b)


@dispatch(ZeroType, object)
def stretch(a, b): return a


@dispatch(StretchedType, object)
def stretch(a, b): return stretch(a[0], a.extent * b)


# Equality:

@dispatch(object, object)
def equal(a, b): return False


@dispatch(PrimitiveType, PrimitiveType)
def equal(a, b): return type(a) == type(b)


@dispatch.multi((SumType, SumType),
                (ProductType, ProductType))
def equal(a, b): return (equal(a[0], b[0]) and equal(a[1], b[1])) or \
                        (equal(a[0], b[1]) and equal(a[1], b[0]))
