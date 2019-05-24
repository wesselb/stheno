# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import operator

import numpy as np
import pytest
from plum import NotFoundLookupError

from stheno.field import (
    mul,
    add,
    SumElement,
    new,
    get_field,
    broadcast,
    WrappedElement,
    JoinElement,
    Element,
    OneElement,
    ZeroElement,
    ScaledElement,
    ProductElement
)
from stheno.function_field import (
    stretch,
    differentiate,
    shift,
    transform,
    select,
    Function,
    StretchedFunction,
    ShiftedFunction,
    SelectedFunction,
    InputTransformedFunction,
    DerivativeFunction,
    TensorProductFunction,
    OneFunction,
    ZeroFunction,
    _to_list,
    tuple_equal
)
from stheno.function_field import to_tensor
from stheno.kernel import (
    EQ,
    RQ,
    Linear,
    OneKernel,
    ZeroKernel,
    Delta,
    TensorProductKernel,
    Kernel
)
from stheno.matrix import Dense
from stheno.mean import TensorProductMean, ZeroMean, OneMean, Mean
from .util import allclose


def test_corner_cases():
    with pytest.raises(IndexError):
        EQ().stretch(1)[1]
    with pytest.raises(IndexError):
        (EQ() + RQ(1))[2]
    with pytest.raises(RuntimeError):
        mul(1, 1)
    with pytest.raises(RuntimeError):
        add(1, 1)
    with pytest.raises(RuntimeError):
        stretch(1, 1)
    with pytest.raises(RuntimeError):
        transform(1, 1)
    with pytest.raises(RuntimeError):
        differentiate(1, 1)
    with pytest.raises(RuntimeError):
        select(1, 1)
    with pytest.raises(RuntimeError):
        shift(1, 1)
    assert repr(EQ()) == str(EQ())
    assert EQ().__name__ == 'EQ'
    with pytest.raises(NotImplementedError):
        WrappedElement(1).display(1, lambda x: x)
    with pytest.raises(NotImplementedError):
        JoinElement(1, 2).display(1, 2, lambda x: x)


def test_to_tensor():  # From function field.
    assert isinstance(to_tensor(np.array([1, 2])), np.ndarray)
    assert isinstance(to_tensor([1, 2]), np.ndarray)


def test_tuple_equal():
    assert tuple_equal((1,), (1,))
    assert tuple_equal((1, [1]), (1, [1]))
    assert not tuple_equal((1,), ([1],))
    assert not tuple_equal(([1],), ([1], [1]))
    assert tuple_equal(([1],), ([1],))
    assert not tuple_equal(([1],), ([2],))
    assert not tuple_equal(([1], [1, 2]), ([1, 2], [1, 2]))
    assert not tuple_equal(([1], [1, 2]), ([1], [1, 3]))


def test_stretch_shorthand():
    assert str(EQ() > 2) == str(EQ().stretch(2))


def test_equality_field():
    one, zero = OneElement(), ZeroElement()

    assert Element() != Element()

    # Ones and zeros:
    assert one == one
    assert one != zero
    assert zero == zero

    # Scaled elements:
    assert ScaledElement(one, 1) == ScaledElement(one, 1)
    assert ScaledElement(one, 2) != ScaledElement(one, 1)
    assert ScaledElement(zero, 1) != ScaledElement(one, 1)

    # Product elements:
    assert ProductElement(one, zero) == ProductElement(one, zero)
    assert ProductElement(one, zero) == ProductElement(zero, one)
    assert ProductElement(one, zero) != ProductElement(one, one)

    # Sum elements:
    assert SumElement(one, zero) == SumElement(one, zero)
    assert SumElement(one, zero) == SumElement(zero, one)
    assert SumElement(one, zero) != SumElement(one, one)


def test_equality_function_field():
    one, zero = OneFunction(), ZeroFunction()

    def f1(x):
        return x

    def f2(x):
        return x ** 2

    # Test wrapped elements.
    for Type, val1, val2 in [(StretchedFunction, 1, 2),
                             (ShiftedFunction, 1, 2),
                             (SelectedFunction, (1,), (2,)),
                             (InputTransformedFunction, f1, f2),
                             (DerivativeFunction, 1, 2)]:
        assert Type(one, val1) == Type(one, val1)
        assert Type(one, val1) != Type(one, val2)
        assert Type(one, val1) != Type(zero, val1)

    assert TensorProductFunction(f1) == TensorProductFunction(f1)
    assert TensorProductFunction(f1) != TensorProductFunction(f2)


def test_equality_integration():
    # This can be more thorough.
    assert EQ() + EQ() == 2 * EQ()
    assert EQ() != EQ() + 1
    assert EQ().stretch(1).select(0) == EQ().stretch(1).select(0)
    assert Linear() - 1 == Linear() - 1


def test_registrations():
    for x in [Function, Kernel, Mean, Dense]:
        assert get_field.invoke(x)(1) == x

    class MyField(Element):
        pass

    # Test registration of fields and creation of new types.
    with pytest.raises(RuntimeError):
        get_field(MyField())
    get_field.extend(MyField)(lambda _: MyField)
    assert get_field(MyField())
    with pytest.raises(RuntimeError):
        new(MyField(), SumElement)


def test_broadcast():
    assert broadcast(operator.add, (1, 2, 3), (2, 3, 4)) == (3, 5, 7)
    assert broadcast(operator.add, (1,), (2, 3, 4)) == (3, 4, 5)
    assert broadcast(operator.add, (1, 2, 3), (2,)) == (3, 4, 5)
    with pytest.raises(ValueError):
        broadcast(operator.add, (1, 2), (1, 2, 3))
    assert str(EQ().stretch(2).stretch(1, 3)) == 'EQ() > (2, 6)'
    assert str(EQ().stretch(1, 3).stretch(2)) == 'EQ() > (2, 6)'
    assert str(Linear().shift(2).shift(1, 3)) == 'Linear() shift (3, 5)'
    assert str(Linear().shift(1, 3).shift(2)) == 'Linear() shift (3, 5)'


def test_subtraction_and_negation():
    assert str(-EQ()) == '-1 * EQ()'
    assert str(EQ() - EQ()) == '0'
    assert str(RQ(1) - EQ()) == 'RQ(1) + -1 * EQ()'
    assert str(1 - EQ()) == '1 + -1 * EQ()'
    assert str(EQ() - 1) == 'EQ() + -1 * 1'


def test_power():
    with pytest.raises(ValueError):
        EQ() ** -1
    with pytest.raises(NotFoundLookupError):
        EQ() ** .5
    assert str(EQ() ** 0) == '1'
    assert str(EQ() ** 1) == 'EQ()'
    assert str(EQ() ** 2) == 'EQ() * EQ()'
    assert str(EQ() ** 3) == 'EQ() * EQ() * EQ()'


def test_cancellations_zero():
    # With constants:
    assert str(1 * EQ()) == 'EQ()'
    assert str(EQ() * 1) == 'EQ()'
    assert str(0 * EQ()) == '0'
    assert str(EQ() * 0) == '0'
    assert str(0 + EQ()) == 'EQ()'
    assert str(EQ() + 0) == 'EQ()'
    assert str(0 + OneMean()) == '1'
    assert str(OneMean() + 0) == '1'

    # Adding to zero:
    assert str(0 + ZeroKernel()) == '0'
    assert str(ZeroKernel() + 0) == '0'
    assert str(1 + ZeroKernel()) == '1'
    assert str(ZeroKernel() + 1) == '1'
    assert str(2 + ZeroKernel()) == '2 * 1'
    assert str(ZeroKernel() + 2) == '2 * 1'

    # Sums:
    assert str(EQ() + EQ()) == '2 * EQ()'
    assert str(ZeroKernel() + EQ()) == 'EQ()'
    assert str(EQ() + ZeroKernel()) == 'EQ()'
    assert str(ZeroKernel() + ZeroKernel()) == '0'

    # Products:
    assert str(EQ() * EQ()) == 'EQ() * EQ()'
    assert str(ZeroKernel() * EQ()) == '0'
    assert str(EQ() * ZeroKernel()) == '0'
    assert str(ZeroKernel() * ZeroKernel()) == '0'

    # Scales:
    assert str(5 * ZeroKernel()) == '0'
    assert str(ZeroKernel() * 5) == '0'
    assert str(EQ() * 5) == '5 * EQ()'
    assert str(5 * EQ()) == '5 * EQ()'

    # Stretches:
    assert str(ZeroKernel().stretch(5)) == '0'
    assert str(EQ().stretch(5)) == 'EQ() > 5'

    # Periodicisations:
    assert str(ZeroKernel().periodic(5)) == '0'
    assert str(EQ().periodic(5)) == 'EQ() per 5'

    # Reversals:
    assert str(reversed(ZeroKernel())) == '0'
    assert str(reversed(EQ())) == 'EQ()'
    assert str(reversed(Linear())) == 'Reversed(Linear())'

    # Integration:
    assert str(EQ() * EQ() + ZeroKernel() * EQ()) == 'EQ() * EQ()'
    assert str(EQ() * ZeroKernel() + ZeroKernel() * EQ()) == '0'


def test_cancellations_one():
    # Products:
    assert str(EQ() * EQ()) == 'EQ() * EQ()'
    assert str(OneKernel() * EQ()) == 'EQ()'
    assert str(EQ() * OneKernel()) == 'EQ()'
    assert str(OneKernel() * OneKernel()) == '1'


def test_grouping():
    # Scales:
    assert str(5 * EQ()) == '5 * EQ()'
    assert str(5 * (5 * EQ())) == '25 * EQ()'

    # Stretches:
    assert str(EQ().stretch(5)) == 'EQ() > 5'
    assert str(EQ().stretch(5).stretch(5)) == 'EQ() > 25'

    # Shifts:
    assert str(Linear().shift(5)) == 'Linear() shift 5'
    assert str(Linear().shift(5).shift(5)) == 'Linear() shift 10'

    # Products:
    assert str((5 * EQ()) * (5 * EQ())) == '25 * EQ()'
    assert str((5 * (EQ() * EQ())) * (5 * EQ() * EQ())) == '25 * EQ() * EQ()'
    assert str((5 * RQ(1)) * (5 * RQ(2))) == '25 * RQ(1) * RQ(2)'

    # Sums:
    assert str((5 * EQ()) + (5 * EQ())) == '10 * EQ()'
    assert str(EQ() + (5 * EQ())) == '6 * EQ()'
    assert str((5 * EQ()) + EQ()) == '6 * EQ()'
    assert str((EQ() + EQ())) == '2 * EQ()'
    assert str((5 * (EQ() * EQ())) + (5 * (EQ() * EQ()))) == '10 * EQ() * EQ()'
    assert str((5 * RQ(1)) + (5 * RQ(2))) == '5 * RQ(1) + 5 * RQ(2)'

    # Reversal:
    assert str(reversed(Linear() + EQ())) == 'Reversed(Linear()) + EQ()'
    assert str(reversed(Linear() * EQ())) == 'Reversed(Linear()) * EQ()'


def test_parentheses():
    assert str((reversed(Linear() * Linear() +
                         2 * EQ().stretch(1).periodic(2) +
                         RQ(3).periodic(4))) * (EQ().stretch(1) + EQ())) == \
           '(Reversed(Linear()) * Reversed(Linear()) + ' \
           '2 * ((EQ() > 1) per 2) + ' \
           'RQ(3) per 4) * (EQ() > 1 + EQ())'


def test_terms():
    k = EQ() + EQ() * Linear() + RQ(1) * RQ(2) + Delta()
    assert k.num_terms == 4
    assert str(k.term(0)) == 'EQ()'
    assert str(k.term(1)) == 'EQ() * Linear()'
    assert str(k.term(2)) == 'RQ(1) * RQ(2)'
    assert str(k.term(3)) == 'Delta()'
    with pytest.raises(IndexError):
        k.term(4)
    with pytest.raises(IndexError):
        EQ().term(1)


def test_factors():
    k = EQ() * Linear()
    assert k.num_factors == 2
    assert str(k.factor(0)) == 'EQ()'
    assert str(k.factor(1)) == 'Linear()'
    with pytest.raises(IndexError):
        k.factor(2)

    k = (EQ() + EQ()) * Delta() * (RQ(1) + Linear())
    assert k.num_factors == 4
    assert str(k.factor(0)) == '2'
    assert str(k.factor(1)) == 'EQ()'
    assert str(k.factor(2)) == 'Delta()'
    assert str(k.factor(3)) == 'RQ(1) + Linear()'
    with pytest.raises(IndexError):
        k.factor(4)
    with pytest.raises(IndexError):
        EQ().factor(1)


def test_indexing():
    assert str((5 * EQ())[0]) == 'EQ()'
    assert str((EQ() + RQ(1.0))[0]) == 'EQ()'
    assert str((EQ() + RQ(1.0))[1]) == 'RQ(1.0)'
    assert str((EQ() * RQ(1.0))[0]) == 'EQ()'
    assert str((EQ() * RQ(1.0))[1]) == 'RQ(1.0)'


def test_shifting():
    # Kernels:
    assert str(ZeroKernel().shift(5)) == '0'
    assert str(EQ().shift(5)) == 'EQ()'
    assert str(Linear().shift(5)) == 'Linear() shift 5'
    assert str((5 * EQ()).shift(5)) == '5 * EQ()'
    assert str((5 * Linear()).shift(5)) == '(5 * Linear()) shift 5'

    # Means:
    def mean(x): return x

    m = TensorProductMean(mean)
    assert str(ZeroMean().shift(5)) == '0'
    assert str(m.shift(5)) == 'mean shift 5'
    assert str(m.shift(5).shift(5)) == 'mean shift 10'
    assert str((5 * m).shift(5)) == '(5 * mean) shift 5'


def test_selection():
    allclose(_to_list((1, 2)), [1, 2])
    allclose(_to_list([1, 2]), [1, 2])
    allclose(_to_list(np.array([1, 2])), [1, 2])
    allclose(_to_list(1), [1])
    with pytest.raises(ValueError):
        _to_list(np.ones((1, 1)))

    assert str(EQ().select(0)) == 'EQ() : [0]'
    assert str(EQ().select([0, 2])) == 'EQ() : [0, 2]'
    assert str(EQ().select(0, 2)) == 'EQ() : ([0], [2])'
    assert str(EQ().select([0], 2)) == 'EQ() : ([0], [2])'
    assert str(EQ().select(0, [2])) == 'EQ() : ([0], [2])'
    assert str(EQ().select([0, 1], [2])) == 'EQ() : ([0, 1], [2])'
    assert str(EQ().select(None, [2])) == 'EQ() : (None, [2])'
    assert str(EQ().select(None)) == 'EQ() : None'
    assert str(EQ().select(None, None)) == 'EQ() : (None, None)'
    assert str(EQ().select([1], None)) == 'EQ() : ([1], None)'

    assert str(ZeroKernel().select(0)) == '0'
    assert str(OneMean().select(0)) == '1'


def test_input_transform():
    assert str(EQ().transform(lambda x: x)) == 'EQ() transform <lambda>'
    assert str(EQ().transform(lambda x: x, lambda x: x)) == \
           'EQ() transform (<lambda>, <lambda>)'

    assert str(ZeroKernel().transform(lambda x: x)) == '0'
    assert str(OneMean().transform(lambda x: x)) == '1'


def test_derivative():
    assert str(EQ().diff(0)) == 'd(0) EQ()'
    assert str(EQ().diff(0, 1)) == 'd(0, 1) EQ()'

    assert str(ZeroKernel().diff(0)) == '0'
    assert str(OneKernel().diff(0)) == '0'

    assert str(ZeroMean().diff(0)) == '0'
    assert str(OneMean().diff(0)) == '0'


def test_function():
    def f():
        pass

    assert str(ZeroKernel() * f) == '0'
    assert str(f * ZeroKernel()) == '0'
    assert str(OneKernel() * f) == 'f'
    assert str(f * OneKernel()) == 'f'
    assert str(EQ() * f) == 'EQ() * f'
    assert str(f * EQ()) == 'f * EQ()'
    assert str((EQ() * EQ()) * f) == 'EQ() * EQ() * f'
    assert str(f * (EQ() * EQ())) == 'f * EQ() * EQ()'
    assert str((5 * EQ()) * f) == '5 * EQ() * f'
    assert str(f * (5 * EQ())) == '5 * f * EQ()'

    assert str(ZeroKernel() + f) == 'f'
    assert str(f + ZeroKernel()) == 'f'
    assert str(OneKernel() + f) == '1 + f'
    assert str(f + OneKernel()) == 'f + 1'
    assert str(EQ() + f) == 'EQ() + f'
    assert str(f + EQ()) == 'f + EQ()'
    assert str((EQ() + RQ(1)) + f) == 'EQ() + RQ(1) + f'
    assert str(f + (EQ() + RQ(1))) == 'f + EQ() + RQ(1)'
    assert str((5 + EQ()) + f) == '5 * 1 + EQ() + f'
    assert str(f + (5 + EQ())) == 'f + 5 * 1 + EQ()'

    assert str(OneKernel() * f) == 'f'
    assert str(OneKernel() * TensorProductKernel((lambda x, c: x), f)) == \
           '(<lambda> x f)'
    assert str(OneKernel() * TensorProductKernel(f, (
        lambda x, c: x))) == '(f x <lambda>)'


def test_formatting():
    assert (2 * EQ().stretch(3)).display(lambda x: x ** 2) == '4 * (EQ() > 9)'
