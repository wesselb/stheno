from lab import B
from matrix import AbstractMatrix, Diagonal
from mlkernels import PosteriorKernel, PosteriorMean, SubspaceKernel, num_elements
from plum import Union

from .. import _dispatch
from .fdd import FDD
from .gp import cross

__all__ = [
    "combine",
    "AbstractObservations",
    "AbstractPseudoObservations",
    "Observations",
    "Obs",
    "PseudoObservations",
    "SparseObservations",
    "PseudoObs",
    "SparseObs",
    "PseudoObservationsFITC",
    "PseudoObsFITC",
    "PseudoObservationsDTC",
    "PseudoObsDTC",
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
    def __init__(self, fdd: FDD, y: Union[B.Numeric, AbstractMatrix]):
        y_shape = B.shape(y)  # Save shape for error message.
        y = B.uprank(y, rank=2)

        if not B.shape(y, -1) == 1:
            raise ValueError(f"Invalid shape of observed values {y_shape}.")

        # Handle missing data.
        available = B.jit_to_numpy(~B.isnan(y[:, 0]))
        if not B.all(available):
            fdd = B.take(fdd, available)
            y = B.take(y, available)

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


class AbstractPseudoObservations(AbstractObservations):
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
        AbstractPseudoObservations.__init__(self, combine(*us), *args)

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

    def posterior_kernel(self, measure, p_i, p_j):
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
            self.A(measure),
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

        L_z = B.cholesky(K_z)
        iLz_Kzx = B.solve(L_z, K_zx)
        A = B.add(B.eye(K_z), B.iqf(K_n, B.transpose(iLz_Kzx)))

        if self.method in {"vfe", "fitc"}:
            K_x_diag = measure.kernels[p_x].elwise(x)[..., 0]
            Q_x_diag = B.matmul_diag(iLz_Kzx, iLz_Kzx, tr_a=True)
            diag_correction = Diagonal(K_x_diag) - Diagonal(Q_x_diag)

        if self.method == "vfe":
            K_n += 0
            trace_part = B.ratio(diag_correction, K_n)
        elif self.method == "fitc":
            K_n += diag_correction
            trace_part = 0
        elif self.method == "dtc":
            K_n += 0
            trace_part = 0
        else:  # pragma: no cover
            # This cannot be reached.
            raise ValueError(f'Invalid approximation method "{method}".')

        # Subspace variance:
        self._A_store[id(measure)] = B.mm(L_z, A, L_z, tr_c=True)

        # Optimal mean:
        y_bar = B.subtract(B.uprank(self.y), measure.means[p_x](x))
        prod_y_bar = B.iqf(K_n, B.transpose(iLz_Kzx), y_bar)
        # TODO: Absorb `L_z` in the posterior mean for better stability.
        mu = B.add(measure.means[p_z](z), B.iqf(A, B.transpose(L_z), prod_y_bar))
        self._mu_store[id(measure)] = mu

        # Compute the ELBO.
        dtype = B.dtype_float(K_n)
        det_part = B.logdet(B.multiply(B.cast(dtype, 2 * B.pi), K_n)) + B.logdet(A)
        iqf_part = B.iqf_diag(K_n, y_bar)[..., 0] - B.iqf_diag(A, prod_y_bar)[..., 0]
        self._elbo_store[id(measure)] = -0.5 * (det_part + iqf_part + trace_part)


class PseudoObservations(AbstractPseudoObservations):
    """Observations through inducing points with the VFE approximation.

    Titsias M. (2009). "Variational Learning of Inducing Variables in Sparse Gaussian
        Processes," in Artificial Intelligence and Statistics, 12th International
        Conference on.

    Further takes arguments according to the constructor of
    :class:`.measure.AbstractObservations`. Can also take in tuples of inducing points.

    Args:
        u (:class:`.fdd.FDD`): Inducing points

    Attributes:
        method (str): Description of the method.
    """

    @property
    def method(self):
        return "vfe"


class PseudoObservationsFITC(AbstractPseudoObservations):
    """Observations through inducing points with the FITC approximation.

    Snelson G. and Ghahramani Z. (2006). "Sparse Gaussian Processes Using
        Pseudo-Inputs," in Neural Information Processing Systems, 18th.

    Further takes arguments according to the constructor of
    :class:`.measure.AbstractObservations`. Can also take in tuples of inducing points.

    Args:
        u (:class:`.fdd.FDD`): Inducing points

    Attributes:
        method (str): Description of the method.
    """

    @property
    def method(self):
        return "fitc"


class PseudoObservationsDTC(AbstractPseudoObservations):
    """Observations through inducing points with the DTC approximation.

    Csato L. and Opper M. (2002). "Sparse On-Line Gaussian Processes," in Neural
        Computation 14 (3), 641-668.

    Seeger, M., Williams, C. K. I., Lawrence, N. D. (2003). "Fast Forward Selection to
        Speed Up Sparse Gaussian Process Regression," in Artificial Intelligence and
        Statistics, 9th International Workshop on.

    Further takes arguments according to the constructor of
    :class:`.measure.AbstractObservations`. Can also take in tuples of inducing points.

    Args:
        u (:class:`.fdd.FDD`): Inducing points

    Attributes:
        method (str): Description of the method.
    """

    @property
    def method(self):
        return "dtc"


Obs = Observations  #: Shorthand for `Observations`.
PseudoObs = PseudoObservations  #: Shorthand for `PseudoObservations`.
PseudoObsFITC = PseudoObservationsFITC  #: Shorthand for `PseudoObservationsFITC`.
PseudoObsDTC = PseudoObservationsDTC  #: Shorthand for `PseudoObservationsDTC`.

# Backward compatibility:
SparseObs = PseudoObservations
SparseObservations = PseudoObservations
