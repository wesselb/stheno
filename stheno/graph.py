# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from lab import B
from plum import Dispatcher, Self, Referentiable, type_parameter, Kind, \
    PromisedType

from itertools import product
from stheno import GPPrimitive, ZeroKernel, PosteriorCrossMean, \
    PosteriorCrossKernel, SPD, ZeroMean, Mean, Kernel, Random

__all__ = ['GP', 'model', 'Graph', 'At', 'Means', 'Kernels']

At = Kind


class StorageByID(Referentiable):
    """A dictionary-like type that stores objects by their `id`s."""
    dispatch = Dispatcher(in_class=Self)

    def __init__(self):
        self._store = {}

    @dispatch(object)
    def _resolve_item(self, item):
        return id(item)

    @dispatch(int)
    def _resolve_item(self, item):
        return item

    def __repr__(self):
        return self._store.__repr__()

    def __str__(self):
        return self._store.__str__()

    def __getitem__(self, item):
        try:
            return self._store[self._resolve_item(item)]
        except KeyError:
            self._store[item] = self._build(item)
            return self._store[item]

    def __setitem__(self, item, mean):
        self._store[self._resolve_item(item)] = mean

    def _build(self, item):
        raise KeyError('Cannot build object for item "{}".'.format(item))


class Means(StorageByID):
    """A dictionary-like type to store mean functions indexed by objects."""


class Kernels(StorageByID):
    """A dictionary-like type to store kernels functions indexed by
    pairs of objects.
    """

    def _resolve_item(self, ps):
        if type(ps) is not tuple:
            ps = ps, ps
        return StorageByID._resolve_item(self, ps[0]), \
               StorageByID._resolve_item(self, ps[1])

    def __getitem__(self, ps):
        pi, pj = self._resolve_item(ps)
        try:
            return self._store[pi, pj]
        except KeyError:
            pass

        # Kernel `k_ij` cannot be found. Try looking for `k_ji` and reverse
        # its inputs.
        try:
            return reversed(self._store[pj, pi])
        except KeyError:
            pass

        # Finally, try building the kernel.
        self._store[pi, pj] = self._build((pi, pj))
        return self._store[pi, pj]


class PosteriorMeans(Means):
    """A posterior version of :class:`.graph.Means`.

    Args:
        prior_kernels (instance of :class:`.graph.Kernels`): Prior kernels.
        prior_means (instance of :class:`.graph.Means`): Prior means.
        p_data (instance of :class:`.graph.GP`): Process corresponding to the
            data.
        x (design matrix): Locations of observations.
        Kx (matrix): Kernel matrix of observations.
        y (matrix): Observations.
    """

    def __init__(self, prior_kernels, prior_means, p_data, x, Kx, y):
        Means.__init__(self)
        self.prior_kernels = prior_kernels
        self.prior_means = prior_means
        self.p_data = p_data
        self.x = x
        self.Kx = Kx
        self.y = y

    def _build(self, p):
        return PosteriorCrossMean(
            self.prior_means[p],
            self.prior_means[self.p_data],
            self.prior_kernels[self.p_data, p],
            self.x, self.Kx, self.y
        )


class PosteriorKernels(Kernels):
    """A posterior version of :class:`.graph.Kernels`.

    Args:
        prior_kernels (instance of :class:`.graph.Kernels`): Prior kernels.
        p_data (instance of :class:`.graph.GP`): Process corresponding to the
            data.
        x (design matrix): Locations of observations.
        Kx (matrix): Kernel matrix of observations.
    """

    def __init__(self, prior_kernels, p_data, x, Kx):
        Kernels.__init__(self)
        self.prior_kernels = prior_kernels
        self.p_data = p_data
        self.x = x
        self.Kx = Kx

    def _build(self, ps):
        pi, pj = ps
        return PosteriorCrossKernel(
            self.prior_kernels[pi, pj],
            self.prior_kernels[self.p_data, pi],
            self.prior_kernels[self.p_data, pj],
            self.x, self.Kx
        )


PromisedGP = PromisedType()


class Graph(Referentiable):
    """A GP model."""
    dispatch = Dispatcher(in_class=Self)

    def __init__(self):
        self.ps = []
        self.kernels = Kernels()
        self.means = Means()
        self.prior_kernels = None
        self.prior_means = None

    def add_independent_gp(self, p, kernel, mean):
        """Add an independent GP to the model.

        Args:
            p (instance of :class:`.graph.GP`): GP object to add.
            kernel (instance of :class:`.kernel.Kernel`): Kernel function of GP.
            mean (instance of :class:`.mean.Mean`): Mean function of GP.
        """
        mean = ZeroMean() if mean is None else mean
        self.means[p] = mean
        self.kernels[p] = kernel
        for pi in self.ps:
            self.kernels[p, pi] = ZeroKernel()
        self.ps.append(p)

    @dispatch(object, PromisedGP)
    def sum(self, other, p):
        """Sum a GP from the graph with another object.

        Args:
            obj1 (other object or instance of :class:`.graph.GP`): First term
                in the sum.
            obj2 (other object or instance of :class:`.graph.GP`): Second term
                in the sum.

        Returns:
            The GP corresponding to the sum.
        """
        return self.sum(p, other)

    @dispatch(PromisedGP, object)
    def sum(self, p, other):
        p_sum = GP(self)
        self.means[p_sum] = self.means[p] + other
        self.ps.append(p_sum)
        for pi in self.ps:
            self.kernels[pi, p_sum] = self.kernels[pi, p]
        return p_sum

    @dispatch(PromisedGP, PromisedGP)
    def sum(self, p1, p2):
        p_sum = GP(self)
        self.means[p_sum] = self.means[p1] + self.means[p2]
        self.ps.append(p_sum)
        for pi in self.ps:
            self.kernels[p_sum, pi] = self.kernels[p1, pi] + \
                                      self.kernels[p2, pi]
        return p_sum

    def mul(self, p, other):
        """Multiply a GP from the graph with another object.

        Args:
            p (instance of :class:`.graph.GP`): GP in the product.
            other (obj): Other object in the product.

        Returns:
            The GP corresponding to the product.
        """
        p_prod = GP(self)
        self.means[p_prod] = other * self.means[p]
        self.kernels[p_prod] = other ** 2 * self.kernels[p]
        for pi in self.ps:
            self.kernels[p_prod, pi] = other * self.kernels[p, pi]
        self.ps.append(p_prod)
        return p_prod

    @dispatch(At, object)
    def condition(self, x, y):
        """Condition the graph on data.

        Args:
            x (design matrix): Locations of points to condition on.
            y (design matrix): Observations to condition on.
        """
        p_data, x = type_parameter(x), x.get()
        Kx = SPD(B.reg(self.kernels[p_data](x)))

        # Store prior if it isn't already.
        if self.prior_kernels is None:
            self.prior_kernels = self.kernels
            self.prior_means = self.means

        prior_kernels = self.kernels
        prior_means = self.means
        self.kernels = PosteriorKernels(prior_kernels, p_data, x, Kx)
        self.means = PosteriorMeans(
            prior_kernels, prior_means, p_data, x, Kx, y
        )

    def revert_prior(self):
        """Revert the model back to the state before any conditioning
        operations.
        """
        if self.prior_kernels is not None:
            self.kernels = self.prior_kernels
            self.means = self.prior_means


class GraphMean(Mean, Referentiable):
    """Mean that evaluates to the right mean for a GP attached to a graph.

    Args:
        graph (instance of :class:`.graph.Graph`): Corresponding graph.
        p (instance of :class:`.graph.GP`): Corresponding GP object.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, graph, p):
        self.p = p
        self.graph = graph

    @dispatch(object)
    def __call__(self, x):
        return self.graph.means[self.p](x)

    @dispatch(At)
    def __call__(self, x):
        return self.graph.means[type_parameter(x)](x.get())


class GraphKernel(Kernel, Referentiable):
    """Kernel that evaluates to the right kernel for a GP attached to a graph.

    Args:
        graph (instance of :class:`.graph.Graph`): Corresponding graph.
        p (instance of :class:`.graph.GP`): Corresponding GP object.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, graph, p):
        self.p = p
        self.graph = graph

    @dispatch(object)
    def __call__(self, x):
        return self(x, x)

    @dispatch(object, object)
    def __call__(self, x, y):
        return self.graph.kernels[self.p, self.p](x, y)

    @dispatch(object, At)
    def __call__(self, x, y):
        return self.graph.kernels[self.p, type_parameter(y)](x, y.get())

    @dispatch(At, object)
    def __call__(self, x, y):
        return self.graph.kernels[type_parameter(x), self.p](x.get(), y)

    @dispatch(At, At)
    def __call__(self, x, y):
        return self.graph.kernels[type_parameter(x),
                                  type_parameter(y)](x.get(), y.get())


model = Graph()  #: A default graph provided for convenience


class GP(GPPrimitive, Referentiable):
    """Gaussian process.

    Args:
        kernel (instance of :class:`.kernel.Kernel`): Kernel of the
            process.
        mean (instance of :class:`.mean.Mean`, optional): Mean function of the
            process. Defaults to zero.
        graph (instance of :class:`.graph.Graph`, optional): Graph to attach to.
    """
    dispatch = Dispatcher(in_class=Self)

    @dispatch([object])
    def __init__(self, kernel, mean=None, graph=model):
        GP.__init__(self, graph)
        graph.add_independent_gp(self, kernel, mean)

    @dispatch(Graph)
    def __init__(self, graph):
        GPPrimitive.__init__(
            self,
            GraphKernel(graph, self),
            GraphMean(graph, self)
        )
        self.graph = graph

    @dispatch(object)
    def __add__(self, other):
        return self.graph.sum(self, other)

    @dispatch(Self)
    def __add__(self, other):
        if self.graph != other.graph:
            raise RuntimeError('Can only add GPs from the same graph.')
        return self.graph.sum(self, other)

    @dispatch(object)
    def __mul__(self, other):
        return self.graph.mul(self, other)

    @dispatch(Random)
    def __mul__(self, other):
        return GPPrimitive.__mul__(self, other)

    @dispatch(At, object)
    def condition(self, x, y):
        """Condition the GP """
        self.graph.condition(x, y)
        return self

    @dispatch(object, object)
    def condition(self, x, y):
        self.graph.condition(At(self)(x), y)
        return self

    def __matmul__(self, other):
        """Alternative to writing `At(self)(other)`"""
        return At(self)(other)

    def revert_prior(self):
        """Revert the model back to the state before any conditioning
        operations.
        """
        self.graph.revert_prior()


PromisedGP.deliver(GP)
