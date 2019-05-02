# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import operator
from plum import NotFoundLookupError

import numpy as np
from stheno.function_field import to_tensor
from stheno.field import mul, add, SumElement, new, get_field, broadcast, \
    WrappedElement, JoinElement, Element, OneElement, ZeroElement, \
    ScaledElement, ProductElement
from stheno.function_field import stretch, differentiate, shift, transform, \
    select, Function, StretchedFunction, ShiftedFunction, SelectedFunction, \
    InputTransformedFunction, DerivativeFunction, TensorProductFunction, \
    OneFunction, ZeroFunction, _to_list, tuple_equal
from stheno.kernel import EQ, RQ, Linear, OneKernel, ZeroKernel, Delta, \
    TensorProductKernel, Kernel
from stheno.matrix import Dense
from stheno.mean import TensorProductMean, ZeroMean, OneMean, Mean

# noinspection PyUnresolvedReferences
from . import eq, ok, raises, neq, assert_allclose


def test_corner_cases():
    yield raises, IndexError, lambda: EQ().stretch(1)[1]
    yield raises, IndexError, lambda: (EQ() + RQ(1))[2]
    yield raises, RuntimeError, lambda: mul(1, 1)
    yield raises, RuntimeError, lambda: add(1, 1)
    yield raises, RuntimeError, lambda: stretch(1, 1)
    yield raises, RuntimeError, lambda: transform(1, 1)
    yield raises, RuntimeError, lambda: differentiate(1, 1)
    yield raises, RuntimeError, lambda: select(1, 1)
    yield raises, RuntimeError, lambda: shift(1, 1)
    yield eq, repr(EQ()), str(EQ())
    yield eq, EQ().__name__, 'EQ'
    yield raises, NotImplementedError, \
          lambda: WrappedElement(1).display(1, lambda x: x)
    yield raises, NotImplementedError, \
          lambda: JoinElement(1, 2).display(1, 2, lambda x: x)


def test_to_tensor():  # From function field.
    yield isinstance, to_tensor(np.array([1, 2])), np.ndarray
    yield isinstance, to_tensor([1, 2]), np.ndarray


def test_tuple_equal():
    yield ok, tuple_equal((1,), (1,))
    yield ok, tuple_equal((1, [1]), (1, [1]))
    yield ok, not tuple_equal((1,), ([1],))
    yield ok, not tuple_equal(([1],), ([1], [1]))
    yield ok, tuple_equal(([1],), ([1],))
    yield ok, not tuple_equal(([1],), ([2],))
    yield ok, not tuple_equal(([1], [1, 2]), ([1, 2], [1, 2]))
    yield ok, not tuple_equal(([1], [1, 2]), ([1], [1, 3]))


def test_stretch_shorthand():
    yield eq, str(EQ() > 2), str(EQ().stretch(2))


def test_equality_field():
    one, zero = OneElement(), ZeroElement()

    yield neq, Element(), Element()

    # Ones and zeros:
    yield eq, one, one
    yield neq, one, zero
    yield eq, zero, zero

    # Scaled elements:
    yield eq, ScaledElement(one, 1), ScaledElement(one, 1)
    yield neq, ScaledElement(one, 2), ScaledElement(one, 1)
    yield neq, ScaledElement(zero, 1), ScaledElement(one, 1)

    # Product elements:
    yield eq, ProductElement(one, zero), ProductElement(one, zero)
    yield eq, ProductElement(one, zero), ProductElement(zero, one)
    yield neq, ProductElement(one, zero), ProductElement(one, one)

    # Sum elements:
    yield eq, SumElement(one, zero), SumElement(one, zero)
    yield eq, SumElement(one, zero), SumElement(zero, one)
    yield neq, SumElement(one, zero), SumElement(one, one)


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
        yield eq, Type(one, val1), Type(one, val1)
        yield neq, Type(one, val1), Type(one, val2)
        yield neq, Type(one, val1), Type(zero, val1)

    yield eq, TensorProductFunction(f1), TensorProductFunction(f1)
    yield neq, TensorProductFunction(f1), TensorProductFunction(f2)


def test_equality_integration():
    # This can be more thorough.
    yield eq, EQ() + EQ(), 2 * EQ()
    yield neq, EQ(), EQ() + 1
    yield eq, EQ().stretch(1).select(0), EQ().stretch(1).select(0)
    yield eq, Linear() - 1, Linear() - 1


def test_registrations():
    for x in [Function, Kernel, Mean, Dense]:
        yield eq, get_field.invoke(x)(1), x

    class MyField(Element):
        pass

    # Test registration of fields and creation of new types.
    yield raises, RuntimeError, lambda: get_field(MyField())
    get_field.extend(MyField)(lambda _: MyField)
    yield (lambda: ok(get_field(MyField())))
    yield raises, RuntimeError, lambda: new(MyField(), SumElement)


def test_broadcast():
    yield eq, broadcast(operator.add, (1, 2, 3), (2, 3, 4)), (3, 5, 7)
    yield eq, broadcast(operator.add, (1,), (2, 3, 4)), (3, 4, 5)
    yield eq, broadcast(operator.add, (1, 2, 3), (2,)), (3, 4, 5)
    yield raises, ValueError, lambda: broadcast(operator.add, (1, 2), (1, 2, 3))
    yield eq, str(EQ().stretch(2).stretch(1, 3)), 'EQ() > (2, 6)'
    yield eq, str(EQ().stretch(1, 3).stretch(2)), 'EQ() > (2, 6)'
    yield eq, str(Linear().shift(2).shift(1, 3)), 'Linear() shift (3, 5)'
    yield eq, str(Linear().shift(1, 3).shift(2)), 'Linear() shift (3, 5)'


def test_subtraction_and_negation():
    yield eq, str(-EQ()), '-1 * EQ()'
    yield eq, str(EQ() - EQ()), '0'
    yield eq, str(RQ(1) - EQ()), 'RQ(1) + -1 * EQ()'
    yield eq, str(1 - EQ()), '1 + -1 * EQ()'
    yield eq, str(EQ() - 1), 'EQ() + -1 * 1'


def test_power():
    yield raises, ValueError, lambda: EQ() ** -1
    yield raises, NotFoundLookupError, lambda: EQ() ** .5
    yield eq, str(EQ() ** 0), '1'
    yield eq, str(EQ() ** 1), 'EQ()'
    yield eq, str(EQ() ** 2), 'EQ() * EQ()'
    yield eq, str(EQ() ** 3), 'EQ() * EQ() * EQ()'


def test_cancellations_zero():
    # With constants:
    yield eq, str(1 * EQ()), 'EQ()'
    yield eq, str(EQ() * 1), 'EQ()'
    yield eq, str(0 * EQ()), '0'
    yield eq, str(EQ() * 0), '0'
    yield eq, str(0 + EQ()), 'EQ()'
    yield eq, str(EQ() + 0), 'EQ()'
    yield eq, str(0 + OneMean()), '1'
    yield eq, str(OneMean() + 0), '1'

    # Adding to zero:
    yield eq, str(0 + ZeroKernel()), '0'
    yield eq, str(ZeroKernel() + 0), '0'
    yield eq, str(1 + ZeroKernel()), '1'
    yield eq, str(ZeroKernel() + 1), '1'
    yield eq, str(2 + ZeroKernel()), '2 * 1'
    yield eq, str(ZeroKernel() + 2), '2 * 1'

    # Sums:
    yield eq, str(EQ() + EQ()), '2 * EQ()'
    yield eq, str(ZeroKernel() + EQ()), 'EQ()'
    yield eq, str(EQ() + ZeroKernel()), 'EQ()'
    yield eq, str(ZeroKernel() + ZeroKernel()), '0'

    # Products:
    yield eq, str(EQ() * EQ()), 'EQ() * EQ()'
    yield eq, str(ZeroKernel() * EQ()), '0'
    yield eq, str(EQ() * ZeroKernel()), '0'
    yield eq, str(ZeroKernel() * ZeroKernel()), '0'

    # Scales:
    yield eq, str(5 * ZeroKernel()), '0'
    yield eq, str(ZeroKernel() * 5), '0'
    yield eq, str(EQ() * 5), '5 * EQ()'
    yield eq, str(5 * EQ()), '5 * EQ()'

    # Stretches:
    yield eq, str(ZeroKernel().stretch(5)), '0'
    yield eq, str(EQ().stretch(5)), 'EQ() > 5'

    # Periodicisations:
    yield eq, str(ZeroKernel().periodic(5)), '0'
    yield eq, str(EQ().periodic(5)), 'EQ() per 5'

    # Reversals:
    yield eq, str(reversed(ZeroKernel())), '0'
    yield eq, str(reversed(EQ())), 'EQ()'
    yield eq, str(reversed(Linear())), 'Reversed(Linear())'

    # Integration:
    yield eq, str(EQ() * EQ() + ZeroKernel() * EQ()), 'EQ() * EQ()'
    yield eq, str(EQ() * ZeroKernel() + ZeroKernel() * EQ()), '0'


def test_cancellations_one():
    # Products:
    yield eq, str(EQ() * EQ()), 'EQ() * EQ()'
    yield eq, str(OneKernel() * EQ()), 'EQ()'
    yield eq, str(EQ() * OneKernel()), 'EQ()'
    yield eq, str(OneKernel() * OneKernel()), '1'


def test_grouping():
    # Scales:
    yield eq, str(5 * EQ()), '5 * EQ()'
    yield eq, str(5 * (5 * EQ())), '25 * EQ()'

    # Stretches:
    yield eq, str(EQ().stretch(5)), 'EQ() > 5'
    yield eq, str(EQ().stretch(5).stretch(5)), 'EQ() > 25'

    # Shifts:
    yield eq, str(Linear().shift(5)), 'Linear() shift 5'
    yield eq, str(Linear().shift(5).shift(5)), 'Linear() shift 10'

    # Products:
    yield eq, str((5 * EQ()) * (5 * EQ())), '25 * EQ()'
    yield eq, str((5 * (EQ() * EQ())) * (5 * EQ() * EQ())), \
          '25 * EQ() * EQ()'
    yield eq, str((5 * RQ(1)) * (5 * RQ(2))), '25 * RQ(1) * RQ(2)'

    # Sums:
    yield eq, str((5 * EQ()) + (5 * EQ())), '10 * EQ()'
    yield eq, str(EQ() + (5 * EQ())), '6 * EQ()'
    yield eq, str((5 * EQ()) + EQ()), '6 * EQ()'
    yield eq, str((EQ() + EQ())), '2 * EQ()'
    yield eq, str((5 * (EQ() * EQ())) + (5 * (EQ() * EQ()))), \
          '10 * EQ() * EQ()'
    yield eq, str((5 * RQ(1)) + (5 * RQ(2))), '5 * RQ(1) + 5 * RQ(2)'

    # Reversal:
    yield eq, str(reversed(Linear() + EQ())), 'Reversed(Linear()) + EQ()'
    yield eq, str(reversed(Linear() * EQ())), 'Reversed(Linear()) * EQ()'


def test_parentheses():
    yield eq, str((reversed(Linear() * Linear() +
                            2 * EQ().stretch(1).periodic(2) +
                            RQ(3).periodic(4))) *
                  (EQ().stretch(1) + EQ())), \
          '(Reversed(Linear()) * Reversed(Linear()) + ' \
          '2 * ((EQ() > 1) per 2) + RQ(3) per 4) * (EQ() > 1 + EQ())'


def test_terms():
    k = EQ() + EQ() * Linear() + RQ(1) * RQ(2) + Delta()
    yield eq, k.num_terms, 4
    yield eq, str(k.term(0)), 'EQ()'
    yield eq, str(k.term(1)), 'EQ() * Linear()'
    yield eq, str(k.term(2)), 'RQ(1) * RQ(2)'
    yield eq, str(k.term(3)), 'Delta()'
    yield raises, IndexError, lambda: k.term(4)
    yield raises, IndexError, lambda: EQ().term(1)


def test_factors():
    k = EQ() * Linear()
    yield eq, k.num_factors, 2
    yield eq, str(k.factor(0)), 'EQ()'
    yield eq, str(k.factor(1)), 'Linear()'
    yield raises, IndexError, lambda: k.factor(2)

    k = (EQ() + EQ()) * Delta() * (RQ(1) + Linear())
    yield eq, k.num_factors, 4
    yield eq, str(k.factor(0)), '2'
    yield eq, str(k.factor(1)), 'EQ()'
    yield eq, str(k.factor(2)), 'Delta()'
    yield eq, str(k.factor(3)), 'RQ(1) + Linear()'
    yield raises, IndexError, lambda: k.factor(4)
    yield raises, IndexError, lambda: EQ().factor(1)


def test_indexing():
    yield eq, str((5 * EQ())[0]), 'EQ()'
    yield eq, str((EQ() + RQ(1.0))[0]), 'EQ()'
    yield eq, str((EQ() + RQ(1.0))[1]), 'RQ(1.0)'
    yield eq, str((EQ() * RQ(1.0))[0]), 'EQ()'
    yield eq, str((EQ() * RQ(1.0))[1]), 'RQ(1.0)'


def test_shifting():
    # Kernels:
    yield eq, str(ZeroKernel().shift(5)), '0'
    yield eq, str(EQ().shift(5)), 'EQ()'
    yield eq, str(Linear().shift(5)), 'Linear() shift 5'
    yield eq, str((5 * EQ()).shift(5)), '5 * EQ()'
    yield eq, str((5 * Linear()).shift(5)), '(5 * Linear()) shift 5'

    # Means:
    def mean(x): return x

    m = TensorProductMean(mean)
    yield eq, str(ZeroMean().shift(5)), '0'
    yield eq, str(m.shift(5)), 'mean shift 5'
    yield eq, str(m.shift(5).shift(5)), 'mean shift 10'
    yield eq, str((5 * m).shift(5)), '(5 * mean) shift 5'


def test_selection():
    yield assert_allclose, _to_list((1, 2)), [1, 2]
    yield assert_allclose, _to_list([1, 2]), [1, 2]
    yield assert_allclose, _to_list(np.array([1, 2])), [1, 2]
    yield assert_allclose, _to_list(1), [1]
    yield raises, ValueError, lambda: _to_list(np.ones((1, 1)))

    yield eq, str(EQ().select(0)), 'EQ() : [0]'
    yield eq, str(EQ().select([0, 2])), 'EQ() : [0, 2]'
    yield eq, str(EQ().select(0, 2)), 'EQ() : ([0], [2])'
    yield eq, str(EQ().select([0], 2)), 'EQ() : ([0], [2])'
    yield eq, str(EQ().select(0, [2])), 'EQ() : ([0], [2])'
    yield eq, str(EQ().select([0, 1], [2])), 'EQ() : ([0, 1], [2])'
    yield eq, str(EQ().select(None, [2])), 'EQ() : (None, [2])'
    yield eq, str(EQ().select(None)), 'EQ() : None'
    yield eq, str(EQ().select(None, None)), 'EQ() : (None, None)'
    yield eq, str(EQ().select([1], None)), 'EQ() : ([1], None)'

    yield eq, str(ZeroKernel().select(0)), '0'
    yield eq, str(OneMean().select(0)), '1'


def test_input_transform():
    yield eq, str(EQ().transform(lambda x: x)), 'EQ() transform <lambda>'
    yield eq, str(EQ().transform(lambda x: x, lambda x: x)), \
          'EQ() transform (<lambda>, <lambda>)'

    yield eq, str(ZeroKernel().transform(lambda x: x)), '0'
    yield eq, str(OneMean().transform(lambda x: x)), '1'


def test_derivative():
    yield eq, str(EQ().diff(0)), 'd(0) EQ()'
    yield eq, str(EQ().diff(0, 1)), 'd(0, 1) EQ()'

    yield eq, str(ZeroKernel().diff(0)), '0'
    yield eq, str(OneKernel().diff(0)), '0'

    yield eq, str(ZeroMean().diff(0)), '0'
    yield eq, str(OneMean().diff(0)), '0'


def test_function():
    def f():
        pass

    yield eq, str(ZeroKernel() * f), '0'
    yield eq, str(f * ZeroKernel()), '0'
    yield eq, str(OneKernel() * f), 'f'
    yield eq, str(f * OneKernel()), 'f'
    yield eq, str(EQ() * f), 'EQ() * f'
    yield eq, str(f * EQ()), 'f * EQ()'
    yield eq, str((EQ() * EQ()) * f), 'EQ() * EQ() * f'
    yield eq, str(f * (EQ() * EQ())), 'f * EQ() * EQ()'
    yield eq, str((5 * EQ()) * f), '5 * EQ() * f'
    yield eq, str(f * (5 * EQ())), '5 * f * EQ()'

    yield eq, str(ZeroKernel() + f), 'f'
    yield eq, str(f + ZeroKernel()), 'f'
    yield eq, str(OneKernel() + f), '1 + f'
    yield eq, str(f + OneKernel()), 'f + 1'
    yield eq, str(EQ() + f), 'EQ() + f'
    yield eq, str(f + EQ()), 'f + EQ()'
    yield eq, str((EQ() + RQ(1)) + f), 'EQ() + RQ(1) + f'
    yield eq, str(f + (EQ() + RQ(1))), 'f + EQ() + RQ(1)'
    yield eq, str((5 + EQ()) + f), '5 * 1 + EQ() + f'
    yield eq, str(f + (5 + EQ())), 'f + 5 * 1 + EQ()'

    yield eq, str(OneKernel() * f), 'f'
    yield eq, str(OneKernel() * TensorProductKernel((lambda x, c: x), f)), \
          '(<lambda> x f)'
    yield eq, str(OneKernel() * TensorProductKernel(f, (lambda x, c: x))), \
          '(f x <lambda>)'


def test_formatting():
    yield eq, (2 * EQ().stretch(3)).display(lambda x: x ** 2), '4 * (EQ() > 9)'
