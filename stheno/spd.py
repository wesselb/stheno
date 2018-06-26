# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

from lab import B
from plum import Referentiable, Self, Dispatcher, Kind

__all__ = ['SPD', 'Diagonal', 'UniformDiagonal']


class SPD(Referentiable):
    """Symmetric positive-definite matrix.

    Args:
        mat (tensor): Symmetric positive-definite matrix.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, mat):
        self._mat = mat
        self._cholesky = None
        self._log_det = None
        self._root = None

    @property
    def dtype(self):
        """Data type of the matrix."""
        return B.dtype(self._mat)

    @property
    def mat(self):
        """The matrix."""
        return self._mat

    @property
    def diag(self):
        """Diagonal of the matrix."""
        return B.diag(self._mat)

    @property
    def shape(self):
        """Shape of the matrix."""
        return B.shape(self.mat)

    def cholesky(self):
        """Compute the Cholesky decomposition.
        
        Returns:
            tensor: Cholesky decomposition.
        """
        if self._cholesky is None:
            self._cholesky = B.cholesky(B.reg(self.mat))
        return self._cholesky

    def cholesky_mul(self, a):
        """Multiply the Cholesky decomposition of this matrix with `a`.

        Args:
            a (tensor): Matrix to multiply with.
            
        Returns:
            tensor: Multiplication of the Cholesky of this matrix with `a`.
        """
        return B.matmul(self.cholesky(), a)

    def log_det(self):
        """Compute the log-determinant.
        
        Returns:
            scalar: Log-determinant.
        """
        if self._log_det is None:
            self._log_det = 2 * B.sum(B.log(B.diag(self.cholesky())))
        return self._log_det

    @_dispatch(object, object)
    def mah_dist2(self, a, b, sum=True):
        """Compute the square of the Mahalanobis distance between vectors.

        Args:
            a (tensor): First matrix.
            b (tensor, optional): Second matrix. If omitted, `a` is assumed
                to be the differences.
            sum (bool, optional): Compute the sum of all distances instead
                of returning all distances.
        
        Returns:
            tensor: Distance or distances.
        """
        return self.mah_dist2(a - b, sum=sum)

    @_dispatch(object)
    def mah_dist2(self, diff, sum=True):
        iL_diff = B.trisolve(self.cholesky(), diff)
        return B.sum(iL_diff ** 2) if sum else B.sum(iL_diff ** 2, axis=0)

    def ratio(self, denom):
        """Compute the ratio with respect to another positive-definite matrix.

        Args:
            denom (:class:`.pdmat.PDMat`): Denominator in the ratio.
                
        Returns:
            tensor: Ratio.
        """
        return B.sum(B.trisolve(denom.cholesky(), self.cholesky()) ** 2)

    @_dispatch(object)
    def quadratic_form(self, a):
        """Compute the quadratic form `transpose(a) inv(self.mat) b`.

        Args:
            a (tensor): `a`.
            b (tensor, optional): `b`. Defaults to `a`.
            
        Returns:
            tensor: Quadratic form.
        """
        prod = B.trisolve(self.cholesky(), a)
        return B.dot(prod, prod, tr_a=True)

    @_dispatch(object, object)
    def quadratic_form(self, a, b):
        left = B.trisolve(self.cholesky(), a)
        right = B.trisolve(self.cholesky(), b)
        return B.dot(left, right, tr_a=True)

    @_dispatch(object)
    def quadratic_form_diag(self, a):
        """Compute the diagonal of the quadratic form
        `transpose(a) inv(self.mat) b`.

        Args:
            a (tensor): `a`.
            b (tensor, optional): `b`. Defaults to `a`.
            
        Returns:
            tensor: Diagonal of quadratic form as a rank 1 tensor.
        """
        prod = B.trisolve(self.cholesky(), a)
        return B.sum(prod ** 2, axis=0)

    @_dispatch(object, object)
    def quadratic_form_diag(self, a, b):
        left = B.trisolve(self.cholesky(), a)
        right = B.trisolve(self.cholesky(), b)
        return B.sum(left * right, axis=0)

    def inv_prod(self, a):
        """Compute the matrix-vector product `inv(self.mat) a`.

        Args:
            a (tensor): `a`
            
        Returns:
            tensor: Product.
        """
        return B.cholesky_solve(self.cholesky(), a)

    def root(self):
        """Compute a square root."""
        if self._root is None:
            vals, vecs = B.eig(B.reg(self.mat))
            self._root = B.dot(vecs * vals[None, :] ** .5, vecs, tr_b=True)
        return self._root

    @_dispatch(object)
    def __radd__(self, other):
        return self + other

    @_dispatch(object)
    def __add__(self, other):
        return SPD(self.mat + other)

    @_dispatch(Self)
    def __add__(self, other):
        return SPD(self.mat + other.mat)

    @_dispatch(object)
    def __rmul__(self, other):
        return self * other

    @_dispatch(object)
    def __mul__(self, other):
        if B.is_scalar(other):
            return self * Kind('scalar')(other)
        else:
            raise NotImplementedError('Can only multiply a SPD by '
                                      'a scalar.')

    @_dispatch(Kind('scalar'))
    def __mul__(self, other):
        return SPD(other.get() * self.mat)

    @_dispatch(Self)
    def __mul__(self, other):
        return SPD(self.mat * other.mat)


class Diagonal(SPD, Referentiable):
    """Diagonal symmetric positive-definite matrix.

    Args:
        diag (vector): Diagonal of the matrix.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, diag):
        SPD.__init__(self, None)
        self._diag = diag

    @property
    def dtype(self):
        return B.dtype(self._diag)

    @property
    def diag(self):
        return self._diag

    @property
    def shape(self):
        return B.shape(self.diag)[0], B.shape(self.diag)[0]

    @property
    def mat(self):
        return B.diag(self.diag)

    def cholesky(self):
        return B.diag(self.diag ** .5)

    def cholesky_mul(self, a):
        return a * self.diag[:, None] ** .5

    def log_det(self):
        return B.sum(B.log(self.diag))

    @_dispatch(object)
    def mah_dist2(self, diff, sum=True):
        iL_diff = diff / self.diag[:, None] ** .5
        return B.sum(iL_diff ** 2) if sum else B.sum(iL_diff ** 2, axis=0)

    @_dispatch(Self)
    def ratio(self, denom):
        return B.sum(self.diag / denom.diag)

    @_dispatch(object, object)
    def quadratic_form(self, a, b):
        isqrt_diag = self.diag[:, None] ** .5
        return B.dot(a / isqrt_diag, b / isqrt_diag, tr_a=True)

    @_dispatch(object)
    def quadratic_form(self, a):
        iL_a = a / self.diag[:, None] ** .5
        return B.dot(iL_a, iL_a, tr_a=True)

    @_dispatch(object, object)
    def quadratic_form_diag(self, a, b):
        return B.sum(a * b / self.diag[:, None], axis=0)

    @_dispatch(object)
    def quadratic_form_diag(self, a):
        return B.sum(a ** 2 / self.diag[:, None], axis=0)

    def inv_prod(self, a):
        return a / self.diag[:, None]

    def root(self):
        return self.cholesky()

    @_dispatch(Self)
    def __add__(self, other):
        return Diagonal(self.diag + other.diag)

    @_dispatch(Kind('scalar'))
    def __mul__(self, other):
        return Diagonal(other.get() * self.diag)

    @_dispatch(Self)
    def __mul__(self, other):
        return Diagonal(self.diag * other.diag)


class UniformDiagonal(Diagonal, Referentiable):
    """Uniformly diagonal symmetric positive-definite matrix.

    Args:
        diag_scal (scalar): Scale of the diagonal of the matrix.
        n (int): Size of the the diagonal.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, diag_scale, n):
        Diagonal.__init__(self, None)
        self.diag_scale = diag_scale
        self._n = n

    @property
    def dtype(self):
        return B.dtype(self.diag_scale)

    @property
    def shape(self):
        return self._n, self._n

    @property
    def mat(self):
        return B.eye(self._n, dtype=self.dtype) * self.diag_scale

    @property
    def diag(self):
        return B.ones(self._n, dtype=self.dtype) * self.diag_scale

    def cholesky(self):
        return B.eye(self._n, dtype=self.dtype) * self.diag_scale ** .5

    def cholesky_mul(self, a):
        return a * self.diag_scale ** .5

    def log_det(self):
        return B.cast(self._n, dtype=self.dtype) * B.log(self.diag_scale)

    @_dispatch(object)
    def mah_dist2(self, diff, sum=True):
        iL_diff = diff / self.diag_scale ** .5
        return B.sum(iL_diff ** 2) if sum else B.sum(iL_diff ** 2, axis=0)

    @_dispatch(Self)
    def ratio(self, denom):
        return B.cast(self._n, dtype=self.dtype) \
               * self.diag_scale / denom.diag_scale

    @_dispatch(object, object)
    def quadratic_form(self, a, b):
        return B.dot(a / self.diag_scale ** .5,
                     b / self.diag_scale ** .5, tr_a=True)

    @_dispatch(object)
    def quadratic_form(self, a):
        iL_a = a / self.diag_scale ** .5
        return B.dot(iL_a, iL_a, tr_a=True)

    @_dispatch(object, object)
    def quadratic_form_diag(self, a, b):
        return B.sum(a * b / self.diag_scale, axis=0)

    @_dispatch(object)
    def quadratic_form_diag(self, a):
        return B.sum(a ** 2 / self.diag_scale, axis=0)

    def inv_prod(self, a):
        return a / self.diag_scale

    def root(self):
        return self.cholesky()

    @_dispatch(Self)
    def __add__(self, other):
        return UniformDiagonal(self.diag_scale + other.diag_scale, self._n)

    @_dispatch(Kind('scalar'))
    def __mul__(self, other):
        return UniformDiagonal(other.get() * self.diag_scale, self._n)

    @_dispatch(Self)
    def __mul__(self, other):
        return UniformDiagonal(self.diag_scale * other.diag_scale, self._n)


# In LAB, define addition for SPDs.

@B.add.extend(SPD, object)
def add(spd, other): return spd.__add__(other)


@B.add.extend(object, SPD)
def add(other, spd): return spd.__radd__(other)


@B.add.extend(SPD, SPD)
def add(spd1, spd2): return spd1 + spd2


# In LAB, define multiplication for SPDs.

@B.multiply.extend(SPD, object)
def mul(spd, other): return spd.__mul__(other)


@B.multiply.extend(object, SPD)
def mul(other, spd): return spd.__rmul__(other)


@B.multiply.extend(SPD, SPD)
def mul(spd1, spd2): return spd1 * spd2
