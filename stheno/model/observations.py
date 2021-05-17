from lab import B
from matrix import AbstractMatrix, Diagonal
from mlkernels import (
    num_elements,
    PosteriorKernel,
    SubspaceKernel,
    PosteriorMean,
)
from plum import Union

from .fdd import FDD
from .gp import cross
from .. import _dispatch

__all__ = [
    "combine",
    "AbstractObservations",
    "Observations",
    "Obs",
    "PseudoObservations",
    "PseudoObs",
    "SparseObservations",
    "SparseObs",
]



@_dispatch
def combine(*fdds: FDD):
    """Combine multiple FDDs or tuples of observations into one.

    Args:
        *objs (:class:`.fdd.FDD` or tuple): FDDs or tuples of observations.

    Returns:
        :class:`.fdd.FDD` or tuple: `args` combined into one.
    """
    combined_noise = B.block_diag(*[fdd.noise for fdd in fdds])
    return cross(*[fdd.p for fdd in fdds])(fdds, combined_noise)


@_dispatch
def combine(*pairs: tuple):
    fdds, ys = zip(*pairs)
    combined_fdd = combine(*fdds)
    combined_y = B.concat(*[B.uprank(y) for y in ys], axis=0)
    return combined_fdd, combined_y


class AbstractObservations:
    """Abstract base class for observations.

    Also takes in one or multiple tuples of the described arguments.

    Args:
        fdd (:class:`.fdd.FDD`): FDD of observations.
        y (column vector): Values of observations.

    Attributes:
        fdd (:class:`.fdd.FDD`): FDD of observations.
        y (column vector): Values of observations.
    """

    @_dispatch
    def __init__(self, fdd: Union[FDD], y: Union[B.Numeric, AbstractMatrix]):
        self.fdd = fdd
        self.y = y

    @_dispatch
    def __init__(self, *pairs: tuple):
        AbstractObservations.__init__(self, *combine(*pairs))

    def posterior_kernel(self, measure, p_i, p_j):  # pragma: no cover
        """Get the posterior kernel between two processes.

        Args:
            measure (:class:`.measure.Measure`): Prior.
            p_i (:class:`.gp.GP`): First process.
            p_j (:class:`.gp.GP`): Second process.

        Returns:
            :class:`mlkernels.Kernel`: Posterior kernel between the first and
                second process.
        """
        raise NotImplementedError("Posterior kernel construction not implemented.")

    def posterior_mean(self, measure, p):  # pragma: no cover
        """Get the posterior kernel of a process.

        Args:
            measure (:class:`.measure.Measure`): Prior.
            p (:class:`.gp.GP`): Process.

        Returns:
            :class:`mlkernels.Mean`: Posterior mean of `p`.
        """
        raise NotImplementedError("Posterior mean construction not implemented.")


class Observations(AbstractObservations):
    """Observations.

    Takes arguments according to the constructor of
    :class:`.measure.AbstractObservations`.

    Args:
        fdd (:class:`.fdd.FDD`): FDDs to corresponding to the observations.
        y (tensor): Values of observations.
    """

    def __init__(self, *args):
        AbstractObservations.__init__(self, *args)
        self._K_x_store = {}

    def K_x(self, measure):
        """Kernel matrix of the data.

        Args:
            measure (:class:`.measure.Measure`): Measure.

        Returns:
            matrix: Kernel matrix.
        """
        try:
            return self._K_x_store[id(measure)]
        except KeyError:
            K_x = measure.kernels[self.fdd.p](self.fdd.x) + self.fdd.noise
            self._K_x_store[id(measure)] = K_x
            return K_x

    def posterior_kernel(self, measure, p_i, p_j):
        if num_elements(self.fdd.x) == 0:
            # There are no observations! Just return prior.
            return measure.kernels[p_i, p_j]

        return PosteriorKernel(
            measure.kernels[p_i, p_j],
            measure.kernels[self.fdd.p, p_i],
            measure.kernels[self.fdd.p, p_j],
            self.fdd.x,
            self.K_x(measure),
        )

    def posterior_mean(self, measure, p):
        if num_elements(self.fdd.x) == 0:
            # There are no observations! Just return prior.
            return measure.means[p]

        return PosteriorMean(
            measure.means[p],
            measure.means[self.fdd.p],
            measure.kernels[self.fdd.p, p],
            self.fdd.x,
            self.K_x(measure),
            self.y,
        )


class PseudoObservations(AbstractObservations):
    """Observations through inducing points.

    Further takes arguments according to the constructor of
    :class:`.measure.AbstractObservations`. Can also take in tuples of inducing points.

    Args:
        u (:class:`.fdd.FDD`): Inducing points
    """

    @_dispatch
    def __init__(self, u: FDD, *args):
        AbstractObservations.__init__(self, *args)
        self.u = u
        self._K_z_store = {}
        self._elbo_store = {}
        self._mu_store = {}
        self._A_store = {}

    @_dispatch
    def __init__(self, us: tuple, *args):
        PseudoObservations.__init__(self, combine(*us), *args)

    def K_z(self, measure):
        """Kernel matrix of the data.

        Args:
            measure (:class:`.measure.Measure`): Measure.

        Returns:
            matrix: Kernel matrix.
        """
        try:
            return self._K_z_store[id(measure)]
        except KeyError:
            self._compute(measure)
            return self._K_z_store[id(measure)]

    def elbo(self, measure):
        """ELBO.

        Args:
            measure (:class:`.measure.Measure`): Measure.

        Returns:
            scalar: ELBO.
        """
        try:
            return self._elbo_store[id(measure)]
        except KeyError:
            self._compute(measure)
            return self._elbo_store[id(measure)]

    def mu(self, measure):
        """Mean of optimal approximating distribution.

        Args:
            measure (:class:`.measure.Measure`): Measure.

        Returns:
            matrix: Mean.
        """
        try:
            return self._mu_store[id(measure)]
        except KeyError:
            self._compute(measure)
            return self._mu_store[id(measure)]

    def A(self, measure):
        """Parameter of the corrective variance of the kernel of the optimal
        approximating distribution.

        Args:
            measure (:class:`.measure.Measure`): Measure.

        Returns:
            matrix: Corrective variance.
        """
        try:
            return self._A_store[id(measure)]
        except KeyError:
            self._compute(measure)
            return self._A_store[id(measure)]

    def _compute(self, measure):
        # Extract processes and inputs.
        p_x, x, noise_x = self.fdd.p, self.fdd.x, self.fdd.noise
        p_z, z, noise_z = self.u.p, self.u.x, self.u.noise

        # Construct the necessary kernel matrices.
        K_zx = measure.kernels[p_z, p_x](z, x)
        K_z = B.add(measure.kernels[p_z](z), noise_z)
        self._K_z_store[id(measure)] = K_z

        # Noise kernel matrix:
        K_n = noise_x

        # The approximation can only handle diagonal noise matrices.
        if not isinstance(K_n, Diagonal):
            raise RuntimeError(
                f"Kernel matrix of observation noise must be diagonal, "
                f'not "{type(K_n).__name__}".'
            )

        # And construct the components for the inducing point approximation.
        L_z = B.cholesky(K_z)
        A = B.add(B.eye(K_z), B.iqf(K_n, B.transpose(B.solve(L_z, K_zx))))
        self._A_store[id(measure)] = A
        y_bar = B.subtract(B.uprank(self.y), measure.means[p_x](x))
        prod_y_bar = B.solve(L_z, B.iqf(K_n, B.transpose(K_zx), y_bar))

        # Compute the optimal mean.
        mu = B.add(
            measure.means[p_z](z),
            B.iqf(A, B.solve(L_z, K_z), prod_y_bar),
        )
        self._mu_store[id(measure)] = mu

        # Compute the ELBO.
        # NOTE: The calculation of `trace_part` asserts that `K_n` is diagonal.
        # The rest, however, is completely generic.
        trace_part = B.ratio(
            Diagonal(measure.kernels[p_x].elwise(x)[:, 0])
            - Diagonal(B.iqf_diag(K_z, K_zx)),
            K_n,
        )
        det_part = B.logdet(2 * B.pi * K_n) + B.logdet(A)
        iqf_part = B.iqf(K_n, y_bar)[0, 0] - B.iqf(A, prod_y_bar)[0, 0]
        self._elbo_store[id(measure)] = -0.5 * (trace_part + det_part + iqf_part)

    def posterior_kernel(self, measure, p_i, p_j):
        L = B.chol(self.K_z(measure))
        return PosteriorKernel(
            measure.kernels[p_i, p_j],
            measure.kernels[self.u.p, p_i],
            measure.kernels[self.u.p, p_j],
            self.u.x,
            self.K_z(measure),
        ) + SubspaceKernel(
            measure.kernels[self.u.p, p_i],
            measure.kernels[self.u.p, p_j],
            self.u.x,
            B.mm(L, self.A(measure), L, tr_c=True),
        )

    def posterior_mean(self, measure, p):
        return PosteriorMean(
            measure.means[p],
            measure.means[self.u.p],
            measure.kernels[self.u.p, p],
            self.u.x,
            self.K_z(measure),
            self.mu(measure),
        )


Obs = Observations  #: Shorthand for `Observations`.
PseudoObs = PseudoObservations  #: Shorthand for `PseudoObservations`.

# Backward compatibility:
SparseObs = PseudoObservations
SparseObservations = PseudoObservations
