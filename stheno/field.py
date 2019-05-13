# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from lab import B
from plum import Dispatcher, Referentiable, Self, NotFoundLookupError

from .util import uprank

__all__ = []

_dispatch = Dispatcher()

Formatter = object  #: A formatter can be any object.
priority = 10  #: Priority precedence level.
definite = 20  #: Highest precedence level.


def squeeze(xs):
    """Squeeze an sequence if it only contains a single element.

    Args:
        xs (sequence): Sequence to squeeze.

    Returns:
        object: `xs[0]` if `xs` consists of a single element and `xs` otherwise.
    """
    return xs[0] if len(xs) == 1 else xs


def get_subclasses(c):
    """Get all subclasses of a class.

    Args:
        c (type): Class to get subclasses of.

    Returns:
        list[type]: List of subclasses of `c`.
    """
    return c.__subclasses__() + \
           [x for sc in c.__subclasses__() for x in get_subclasses(sc)]


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


class Element(Referentiable):
    """A field over functions.

    Functions are also referred to as elements of the field. Elements can be
    added and multiplied.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __eq__(self, other):
        return False

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

    @property
    def num_terms(self):
        """Number of terms"""
        return 1

    def term(self, i):
        """Get a specific term.

        Args:
            i (int): Index of term.

        Returns:
            :class:`.field.Element`: The referenced term.
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
            :class:`.field.Element`: The referenced factor.
        """
        if i == 0:
            return self
        else:
            raise IndexError('Index out of range.')

    @property
    def __name__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.display()

    def __str__(self):
        return self.display()

    @_dispatch(Formatter)
    def display(self, formatter):
        """Display the element.

        Args:
            formatter (function, optional): Function to format values.

        Returns:
            str: Element as a string.
        """
        # Due to multiple inheritance, we might arrive here before arriving at a
        # method in the appropriate subclass. The only case to consider is if
        # we're not in a leaf.
        if isinstance(self, (JoinElement, WrappedElement)):
            return pretty_print(self, formatter)
        else:
            return self.__class__.__name__ + '()'

    @_dispatch()
    def display(self):
        return self.display(lambda x: x)


class OneElement(Element, Referentiable):
    """The constant `1`."""
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(Formatter)
    def display(self, formatter):
        return '1'

    @_dispatch(Self)
    def __eq__(self, other):
        return True


class ZeroElement(Element, Referentiable):
    """The constant `0`."""
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(Formatter)
    def display(self, formatter):
        return '0'

    @_dispatch(Self)
    def __eq__(self, other):
        return True


class WrappedElement(Element, Referentiable):
    """A wrapped element.

    Args:
        e (:class:`.field.Element`): Element to wrap.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e):
        self.e = e

    def __getitem__(self, item):
        if item == 0:
            return self.e
        else:
            raise IndexError('Index out of range.')

    @_dispatch(object, Formatter)
    def display(self, e, formatter):
        raise NotImplementedError()


class JoinElement(Element, Referentiable):
    """Two wrapped elements.

    Args:
        e1 (:class:`.field.Element`): First element to wrap.
        e2 (:class:`.field.Element`): Second element to wrap.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e1, e2):
        self.e1 = e1
        self.e2 = e2

    def __getitem__(self, item):
        if item == 0:
            return self.e1
        elif item == 1:
            return self.e2
        else:
            raise IndexError('Index out of range.')

    @_dispatch(object, object, Formatter)
    def display(self, e1, e2, formatter):
        raise NotImplementedError()


class ScaledElement(WrappedElement, Referentiable):
    """Scaled element.

    Args:
        e (:class:`.field.Element`): Element to scale.
        scale (tensor): Scale.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, scale):
        WrappedElement.__init__(self, e)
        self.scale = scale

    @property
    def num_factors(self):
        return self[0].num_factors + 1

    @_dispatch(object, Formatter)
    def display(self, e, formatter):
        return '{} * {}'.format(formatter(self.scale), e)

    def factor(self, i):
        if i >= self.num_factors:
            raise IndexError('Index out of range.')
        else:
            return self.scale if i == 0 else self[0].factor(i - 1)

    @_dispatch(Self)
    def __eq__(self, other):
        return B.all(self.scale == other.scale) and self[0] == other[0]


class ProductElement(JoinElement, Referentiable):
    """Product of elements."""
    _dispatch = Dispatcher(in_class=Self)

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

    @_dispatch(object, object, Formatter)
    def display(self, e1, e2, formatter):
        return '{} * {}'.format(e1, e2)

    @_dispatch(Self)
    def __eq__(self, other):
        return (self[0] == other[0] and self[1] == other[1]) or \
               (self[0] == other[1] and self[1] == other[0])


class SumElement(JoinElement, Referentiable):
    """Sum of elements."""
    _dispatch = Dispatcher(in_class=Self)

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

    @_dispatch(object, object, Formatter)
    def display(self, e1, e2, formatter):
        return '{} + {}'.format(e1, e2)

    @_dispatch(Self)
    def __eq__(self, other):
        return (self[0] == other[0] and self[1] == other[1]) or \
               (self[0] == other[1] and self[1] == other[0])


@_dispatch(object, object)
def mul(a, b):
    """Multiply two elements.

    Args:
        a (:class:`.field.Element`): First element in product.
        b (:class:`.field.Element`): Second element in product.

    Returns:
        :class:`.field.Element`: Product of the elements.
    """
    raise NotImplementedError('Multiplication not implemented for "{}" and '
                              '"{}"'.format(type(a).__name__, type(b).__name__))


@_dispatch(object, object)
def add(a, b):
    """Add two elements.

    Args:
        a (:class:`.field.Element`): First element in addition.
        b (:class:`.field.Element`): Second element in addition.

    Returns:
        :class:`.field.Element`: Sum of the elements.
    """
    raise NotImplementedError('Addition not implemented for "{}" and "{}".'
                              ''.format(type(a).__name__, type(b).__name__))


@_dispatch(object)
def get_field(a):
    """Get the field of an element.

    Args:
        a (:class:`.field.Element`): Element to get field of.

    Returns:
        type: Field of `a`.
    """
    raise RuntimeError('Could not determine field type of "{}".'
                       ''.format(type(a).__name__))


new_cache = {}


def new(a, t):
    """Create a new specialised type.

    Args:
        a (:class:`.field.Element`): Element to create new type for.
        t (type): Type to create.

    Returns:
        type: Specialisation of `t` appropriate for `a`.
    """
    try:
        return new_cache[type(a), t]
    except KeyError:
        field = get_field(a)
        candidates = list(set(get_subclasses(field)) & set(get_subclasses(t)))

        # There should only be a single candidate.
        if len(candidates) != 1:
            raise RuntimeError('Could not determine "{}" for field "{}".'
                               ''.format(t.__name__, field.__name__))

        new_cache[type(a), t] = candidates[0]
        return new_cache[type(a), t]


# Pretty printing with minimal parentheses.

@_dispatch(Element, Formatter)
def pretty_print(el, formatter):
    """Pretty print an element with a minimal number of parentheses.

    Args:
        el (:class:`.field.Element`): Element to print.
        formatter (:class:`.field.Formatter`): Formatter for values.

    Returns:
        str: `el` converted to string prettily.
    """
    return el.display(formatter)


@_dispatch(WrappedElement, Formatter)
def pretty_print(el, formatter):
    return el.display(pretty_print(el[0], el, formatter), formatter)


@_dispatch(JoinElement, Formatter)
def pretty_print(el, formatter):
    return el.display(pretty_print(el[0], el, formatter),
                      pretty_print(el[1], el, formatter), formatter)


@_dispatch(Element, Element, Formatter)
def pretty_print(el, parent, formatter):
    if need_parens(el, parent):
        return '(' + pretty_print(el, formatter) + ')'
    else:
        return pretty_print(el, formatter)


@_dispatch(Element, SumElement)
def need_parens(el, parent):
    """Check whether `el` needs parentheses when printed in `parent`.

    Args:
        el (:class:`.field.Element`): Element to print.
        parent (:class:`.field.Element`): Parent of element to print.

    Returns:
        bool: Boolean whether `el` needs parentheses.
    """
    return False


@_dispatch(Element, ProductElement)
def need_parens(el, parent): return False


@_dispatch({SumElement, WrappedElement}, ProductElement)
def need_parens(el, parent): return True


@_dispatch(ScaledElement, ProductElement)
def need_parens(el, parent): return False


@_dispatch(Element, WrappedElement)
def need_parens(el, parent): return False


@_dispatch({WrappedElement, JoinElement}, WrappedElement)
def need_parens(el, parent): return True


@_dispatch({ProductElement, ScaledElement}, ScaledElement)
def need_parens(el, parent): return False


# Generic multiplication.

@_dispatch(Element, object)
def mul(a, b):
    if b is 0:
        return new(a, ZeroElement)()
    elif b is 1:
        return a
    else:
        return new(a, ScaledElement)(a, b)


@_dispatch(object, Element)
def mul(a, b):
    return mul(b, a)


@_dispatch(Element, Element)
def mul(a, b): return new(a, ProductElement)(a, b)


# Generic addition.

@_dispatch(Element, object)
def add(a, b):
    if b is 0:
        return a
    else:
        return new(a, SumElement)(a, mul(b, new(a, OneElement)()))


@_dispatch(object, Element)
def add(a, b):
    if a is 0:
        return b
    else:
        return new(b, SumElement)(mul(a, new(b, OneElement)()), b)


@_dispatch(Element, Element)
def add(a, b):
    if a == b:
        return mul(2, a)
    else:
        return new(a, SumElement)(a, b)


# Cancel redundant zeros and ones.

@_dispatch(ZeroElement, object, precedence=definite)
def mul(a, b): return a


@_dispatch(object, ZeroElement, precedence=definite)
def mul(a, b): return b


@_dispatch(ZeroElement, ZeroElement, precedence=definite)
def mul(a, b): return a


@_dispatch(OneElement, Element, precedence=priority)
def mul(a, b): return b


@_dispatch(Element, OneElement, precedence=priority)
def mul(a, b): return a


@_dispatch(OneElement, OneElement, precedence=priority)
def mul(a, b): return a


@_dispatch(ZeroElement, object, precedence=definite)
def add(a, b):
    if b is 0:
        return a
    else:
        return mul(new(a, OneElement)(), b)


@_dispatch(object, ZeroElement, precedence=definite)
def add(a, b):
    if a is 0:
        return b
    else:
        return mul(a, new(b, OneElement)())


@_dispatch(ZeroElement, ZeroElement, precedence=definite)
def add(a, b): return a


@_dispatch(Element, ZeroElement, precedence=definite)
def add(a, b): return a


@_dispatch(ZeroElement, Element, precedence=definite)
def add(a, b): return b


# Group factors and terms if possible.

@_dispatch(object, ScaledElement)
def mul(a, b): return mul(b.scale * a, b[0])


@_dispatch(ScaledElement, object)
def mul(a, b): return mul(a.scale * b, a[0])


@_dispatch(ScaledElement, Element)
def mul(a, b): return mul(a.scale, mul(a[0], b))


@_dispatch(Element, ScaledElement)
def mul(a, b): return mul(b.scale, mul(a, b[0]))


@_dispatch(ScaledElement, ScaledElement)
def mul(a, b):
    if a[0] == b[0]:
        return new(a, ScaledElement)(a[0], a.scale * b.scale)
    else:
        scaled = new(a, ScaledElement)(a[0], a.scale * b.scale)
        return new(a, ProductElement)(scaled, b[0])


@_dispatch(ScaledElement, Element)
def add(a, b):
    if a[0] == b:
        return mul(a.scale + 1, b)
    else:
        return new(a, SumElement)(a, b)


@_dispatch(Element, ScaledElement)
def add(a, b):
    if a == b[0]:
        return mul(b.scale + 1, a)
    else:
        return new(a, SumElement)(a, b)


@_dispatch(ScaledElement, ScaledElement)
def add(a, b):
    if a[0] == b[0]:
        return mul(a.scale + b.scale, a[0])
    else:
        return new(a, SumElement)(a, b)
