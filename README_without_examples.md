# [Stheno](https://github.com/wesselb/stheno)

[![CI](https://github.com/wesselb/stheno/workflows/CI/badge.svg?branch=master)](https://github.com/wesselb/stheno/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/stheno/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/stheno?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/stheno)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Stheno is an implementation of Gaussian process modelling in Python. See 
also [Stheno.jl](https://github.com/willtebbutt/Stheno.jl).

* [Nonlinear Regression in 20 Seconds](#nonlinear-regression-in-20-seconds)
* [Installation](#installation)
* [Manual](#manual)
    - [AutoGrad, TensorFlow, PyTorch, or Jax? Your Choice!](#autograd-tensorflow-pytorch-or-jax-your-choice)
    - [Model Design](#model-design)
    - [Finite-Dimensional Distributions](#finite-dimensional-distributions)
    - [Prior and Posterior Measures](#prior-and-posterior-measures)
    - [Inducing Points](#inducing-points)
    - [Mean and Kernel Design](#mean-and-kernel-design)
* [Examples](#examples)
{examples_toc}

## Nonlinear Regression in 20 Seconds

```python
>>> import numpy as np

>>> from stheno import Measure, GP, EQ

>>> x = np.linspace(0, 2, 10)          # Some points to predict at

>>> y = x ** 2                         # Some observations

>>> prior = Measure()                  # Construct a prior.

>>> f = GP(EQ(), measure=prior)        # Define our probabilistic model.

>>> post = prior | (f(x), y)           # Compute the posterior distribution.

>>> post(f).mean(np.array([1, 2, 3]))  # Predict!
<dense matrix: shape=3x1, dtype=float64
 mat=[[1.   ]
      [4.   ]
      [8.483]]>
```

Moar?! Then read on!

## Installation

See [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).
Then simply

```
pip install stheno
```

## Manual

Note: [here](https://wesselb.github.io/stheno) is a nicely rendered and more
readable version of the docs.

### AutoGrad, TensorFlow, PyTorch, or Jax? Your Choice!

```python
from stheno.autograd import GP, EQ
```

```python
from stheno.tensorflow import GP, EQ
```

```python
from stheno.torch import GP, EQ
```

```python
from stheno.jax import GP, EQ
```

### Model Design

The basic building block is a `f = GP(mean=0, kernel, measure=prior)`, which takes
in [a _mean_, a _kernel_](#mean-and-kernel-design), and a _measure_.
The mean and kernel of a GP can be extracted with `f.mean` and `f.kernel`.
The measure should be thought of as a big joint distribution that assigns a mean and
a kernel to every variable `f`.
A measure can be created with `prior = Measure()`.
A GP `f` can have different means and kernels under different measures.
For example, under some _prior_ measure, `f` can have an `EQ()` kernel; but, under some
_posterior_ measure, `f` has a kernel that is determined by the posterior distribution
of a GP.
[We will see later how posterior measures can be constructed.](#prior-and-posterior-measures)
The measure with which a `f = GP(kernel, measure=prior)` is constructed can be
extracted with `f.measure == prior`.
If the keyword argument `measure` is not set, then automatically a new measure is
created, which afterwards can be extracted with `f.measure`.

Definition, where `prior = Measure()`:

```python
f = GP(kernel)

f = GP(mean, kernel)

f = GP(kernel, measure=prior)

f = GP(mean, kernel, measure=prior)
```

GPs that are associated to the same measure can be combined into new GPs, which is
the primary mechanism used to build cool models.

Here's an example model:

```python
>>> prior = Measure()

>>> f1 = GP(lambda x: x ** 2, EQ(), measure=prior)

>>> f1
GP(<lambda>, EQ())

>>> f2 = GP(Linear(), measure=prior)

>>> f2
GP(0, Linear())

>>> f_sum = f1 + f2

>>> f_sum
GP(<lambda>, EQ() + Linear())

>>> f_sum + GP(EQ())  # Not valid: `GP(EQ())` belongs to a new measure!
AssertionError: Processes GP(<lambda>, EQ() + Linear()) and GP(0, EQ()) are associated to different measures.
```

#### Compositional Design

* Add and subtract GPs and other objects.

    Example:
    
    ```python
    >>> GP(EQ(), measure=prior) + GP(Exp(), measure=prior)
    GP(0, EQ() + Exp())

    >>> GP(EQ(), measure=prior) + GP(EQ(), measure=prior)
    GP(0, 2 * EQ())
  
    >>> GP(EQ()) + 1
    GP(1, EQ())
  
    >>> GP(EQ()) + 0
    GP(0, EQ())
  
    >>> GP(EQ()) + (lambda x: x ** 2)
    GP(<lambda>, EQ())

    >>> GP(2, EQ(), measure=prior) - GP(1, EQ(), measure=prior)
    GP(1, 2 * EQ())
    ```
    
* Multiply GPs by other objects.

    Example:
    
    ```python
    >>> 2 * GP(EQ())
    GP(2, 2 * EQ())
  
    >>> 0 * GP(EQ())
    GP(0, 0)

    >>> (lambda x: x) * GP(EQ())
    GP(0, <lambda> * EQ())
    ```
    
* Shift GPs.

    Example:
    
    ```python
    >>> GP(EQ()).shift(1)
    GP(0, EQ() shift 1) 
    ```
    
* Stretch GPs.

    Example:
    
    ```python
    >>> GP(EQ()).stretch(2)
    GP(0, EQ() > 2)
    ```
    
    The `>` operator is implemented to provide a shorthand for stretching:
    
    ```python
    >>> GP(EQ()) > 2
    GP(0, EQ() > 2)
    ```
    
* Select particular input dimensions.

    Example:
    
    ```python
    >>> GP(EQ()).select(1, 3)
    GP(0, EQ() : [1, 3])
    ```
    
    Indexing is implemented to provide a a shorthand for selecting input dimensions:
    
    ```python
    >>> GP(EQ())[1, 3]
    GP(0, EQ() : [1, 3]) 
    ```
    
* Transform the input.

    Example:
    
    ```python
    >>> GP(EQ()).transform(f)
    GP(0, EQ() transform f)
    ```
    
* Numerically take the derivative of a GP.
    The argument specifies which dimension to take the derivative with respect
    to.
    
    Example:
    
    ```python
    >>> GP(EQ()).diff(1)
    GP(0, d(1) EQ())
    ```
    
* Construct a finite difference estimate of the derivative of a GP.
    See `stheno.measure.Measure.diff_approx` for a description of the arguments.
    
    Example:
    
    ```python
    >>> GP(EQ()).diff_approx(deriv=1, order=2)
    GP(50000000.0 * (0.5 * EQ() + 0.5 * ((-0.5 * (EQ() shift (0.0001414213562373095, 0))) shift (0, -0.0001414213562373095)) + 0.5 * ((-0.5 * (EQ() shift (0, 0.0001414213562373095))) shift (-0.0001414213562373095, 0))), 0)
    ```
    
* Construct the Cartesian product of a collection of GPs.

    Example:
    
    ```python
    >>> prior = Measure()

    >>> f1, f2 = GP(EQ(), measure=prior), GP(EQ(), measure=prior)

    >>> cross(f1, f2)
    GP(MultiOutputMean(0, 0), MultiOutputKernel(EQ(), EQ()))
    ```

#### Displaying GPs

Like means and kernels, GPs have a `display` method that accepts a formatter.

Example:

```python
>>> print(GP(2.12345 * EQ()).display(lambda x: '{:.2f}'.format(x)))
GP(2.12 * EQ(), 0)
```

#### Properties of GPs

[Properties of kernels](#properties-of-means-and-kernels) can be queried on GPs directly.

Example:

```python
>>> GP(EQ()).stationary
True
```

#### Naming GPs

It is possible to give a name to a GP.
Names must be strings.
A measure then behaves like a two-way dictionary between GPs and their names.

Example:

```python
>>> prior = Measure()

>>> p = GP(EQ(), name='name', measure=prior)

>>> p.name
'name'

>>> p.name = 'alternative_name'

>>> prior['alternative_name']
GP(0, EQ())

>>> prior[p]
'alternative_name'
```

### Finite-Dimensional Distributions

Simply call a GP to construct a finite-dimensional distribution.
You can then compute the mean, the variance, sample, or compute a logpdf.

Example:

```python
>>> prior = Measure()

>>> f = GP(EQ(), measure=prior)

>>> x = np.array([0., 1., 2.])

>>> f(x)
FDD(GP(0, EQ()), array([0., 1., 2.]))

>>> f(x).mean
array([[0.],
       [0.],
       [0.]])

>>> f(x).var
<dense matrix: shape=3x3, dtype=float64
 mat=[[1.    0.607 0.135]
      [0.607 1.    0.607]
      [0.135 0.607 1.   ]]>
       
>>> y1 = f(x).sample()

>>> y1
array([[-0.45172746],
       [ 0.46581948],
       [ 0.78929767]])
       
>>> f(x).logpdf(y1)
-2.811609567720761

>>> y2 = f(x).sample(2)
array([[-0.43771276, -2.36741858],
       [ 0.86080043, -1.22503079],
       [ 2.15779126, -0.75319405]]

>>> f(x).logpdf(y2)
 array([-4.82949038, -5.40084225])
```

*
    Use `Measure.logpdf` to compute the joint logpdf of multiple observations.

    Definition:

    ```python
    prior.logpdf(f(x), y)

    prior.logpdf((f1(x1), y1), (f2(x2), y2), ...)
    ```
  
*
    Use `Measure.sample` to jointly sample multiple observations.

    Definition, where `prior = Measure()`:

    ```python
    sample = prior.sample(f(x))

    sample1, sample2, ... = prior.sample(f1(x1), f2(x2), ...)
    ```

*
    Use `f(x).marginals()` to efficiently compute the means and 
    the marginal lower and upper 95% central credible region bounds.
    
    Example:

    ```python
    >>> f(x).marginals()
    (array([0., 0., 0.]), array([-1.96, -1.96, -1.96]), array([1.96, 1.96, 1.96]))
    ```

### Prior and Posterior Measures

Conditioning a _prior_ measure on observations gives a _posterior_ measure.
To condition a measure on observations, use `Measure.__or__`.

Definition, where `prior = Measure()` and `f*` and `g*` are `GP`s:

```python
post = prior | (f(x), y)

post = prior | ((f1(x1), y1), (f2(x2), y2), ...)
```

You can then compute a posterior process with `post(f)` and a finite-dimensional
distribution under the posterior with `post(f(x))`.

Let's consider an example.
First, build a model, and sample some values.

```python
>>> prior = Measure()

>>> f = GP(EQ(), measure=prior)

>>> x = np.array([0., 1., 2.])

>>> y = f(x).sample()
```

Then compute the posterior measure.

```python
>>> post = prior | (f(x), y)

>>> post(f)
GP(PosteriorMean(), PosteriorKernel())

>>> post(f).mean(x)
<dense matrix: shape=3x1, dtype=float64
 mat=[[ 0.412]
      [-0.811]
      [-0.933]]>

>>> post(f).kernel(x)
<dense matrix: shape=3x3, dtype=float64
 mat=[[1.e-12 0.e+00 0.e+00]
      [0.e+00 1.e-12 0.e+00]
      [0.e+00 0.e+00 1.e-12]]>

>>> post(f(x))
<stheno.random.Normal at 0x7fa6d7f8c358>

>>> post(f(x)).mean
<dense matrix: shape=3x1, dtype=float64
 mat=[[ 0.412]
      [-0.811]
      [-0.933]]>

>>> post(f(x)).var
<dense matrix: shape=3x3, dtype=float64
 mat=[[1.e-12 0.e+00 0.e+00]
      [0.e+00 1.e-12 0.e+00]
      [0.e+00 0.e+00 1.e-12]]>
```

We can now build on the posterior.

```python
>>> g = GP(Linear(), measure=post)

>>> f_sum = post(f) + g

>>> f_sum
GP(PosteriorMean(), PosteriorKernel() + Linear())
```

However, what we cannot do is mixing the prior and posterior.

```python
>>> f + g
AssertionError: Processes GP(0, EQ()) and GP(0, Linear()) are associated to different measures.
```

### Inducing Points

Stheno supports sparse approximations of posterior distributions.
To construct a sparse approximation, use `Measure.SparseObs`.

Definition:

```python
obs = SparseObs(u(z),  # FDD of inducing points.
                e,     # Independent, additive noise process.
                f(x),  # FDD of observations _without_ the noise process added.
                y)     # Observations.
                
obs = SparseObs(u(z), e, f(x), y)

obs = SparseObs(u(z), (e1, f1(x1), y1), (e2, f2(x2), y2), ...)

obs = SparseObs((u1(z1), u2(z2), ...), e, f(x), y)

obs = SparseObs(u(z), (e1, f1(x1), y1), (e2, f2(x2), y2), ...)

obs = SparseObs((u1(z1), u2(z2), ...), (e1, f1(x1), y1), (e2, f2(x2), y2), ...)
```

Compute the value of the ELBO with `obs.elbo(prior)`, where `prior = Measure()` is
the measure of your models.
Moreover, the posterior measure can be constructed with `prior | obs`.

Let's consider an example.
First, build a model that incorporates noise and sample some observations.

```python
>>> prior = Measure()

>>> f = GP(EQ(), measure=prior)

>>> e = GP(Delta(), measure=prior)

>>> y = f + e

>>> x_obs = np.linspace(0, 10, 2000)

>>> y_obs = y(x_obs).sample()
```

Ouch, computing the logpdf is quite slow:

```python
>>> %timeit y(x_obs).logpdf(y_obs)
219 ms ± 35.7 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

Let's try to use inducing points to speed this up.

```python
>>> x_ind = np.linspace(0, 10, 100)

>>> u = f(x_ind)   # FDD of inducing points.

>>> %timeit SparseObs(u, e, f(x_obs), y_obs).elbo(prior)
9.8 ms ± 181 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

Much better.
And the approximation is good:

```python
>>> SparseObs(u, e, f(x_obs), y_obs).elbo(prior) - y(x_obs).logpdf(y_obs)
-3.537934389896691e-10
```

We can then construct the posterior:

```python
>>> post_approx = prior | SparseObs(u, e, f(x_obs), y_obs)

>>> post_approx(f(x_obs)).mean
<dense matrix: shape=2000x1, dtype=float64
 mat=[[0.469]
      [0.468]
      [0.467]
      ...
      [1.09 ]
      [1.09 ]
      [1.091]]>
```

### Mean and Kernel Design

Inputs to kernels, means, and GPs, henceforth referred to simply as _inputs_, 
must be of one of the following three forms:

* If the input `x` is a _rank 0 tensor_, i.e. a scalar, then `x` refers to a 
single input location. For example, `0` simply refers to the sole input 
location `0`.

* If the input `x` is a _rank 1 tensor_, then every element of `x` is 
interpreted as a separate input location. For example, `np.linspace(0, 1, 10)`
generates 10 different input locations ranging from `0` to `1`.

* If the input `x` is a _rank 2 tensor_, then every _row_ of `x` is 
interpreted as a separate input location. In this case inputs are 
multi-dimensional, and the columns correspond to the various input dimensions.

If `k` is a kernel, say `k = EQ()`, then `k(x, y)` constructs the _kernel 
matrix_ for all pairs of points between `x` and `y`. `k(x)` is shorthand for
`k(x, x)`. Furthermore, `k.elwise(x, y)` constructs the _kernel vector_ pairing
the points in `x` and `y` element wise, which will be a _rank 2 column vector_.

Example:

```python
>>> EQ()(np.linspace(0, 1, 3))
array([[1.        , 0.8824969 , 0.60653066],
       [0.8824969 , 1.        , 0.8824969 ],
       [0.60653066, 0.8824969 , 1.        ]])
 
>>> EQ().elwise(np.linspace(0, 1, 3), 0)
array([[1.        ],
       [0.8824969 ],
       [0.60653066]])
```

Finally, mean functions always output a _rank 2 column vector_.

#### Available Means

Constants function as constant means. Besides that, the following means are 
available:

* `TensorProductMean(f)`:

    $$ m(x) = f(x). $$

    Adding or multiplying a `FunctionType` `f` to or with a mean will 
    automatically translate `f` to `TensorProductMean(f)`. For example,
    `f * m` will translate to `TensorProductMean(f) * m`, and `f + m` will 
    translate to `TensorProductMean(f) + m`.

#### Available Kernels

Constants function as constant kernels. Besides that, the following kernels are 
available:

* `EQ()`, the exponentiated quadratic:

    $$ k(x, y) = \exp\left( -\frac{1}{2}\|x - y\|^2 \right); $$

* `RQ(alpha)`, the rational quadratic:

    $$ k(x, y) = \left( 1 + \frac{\|x - y\|^2}{2 \alpha} \right)^{-\alpha}; $$

* `Exp()` or `Matern12()`, the exponential kernel:

    $$ k(x, y) = \exp\left( -\|x - y\| \right); $$

* `Matern32()`, the Matern–3/2 kernel:

    $$ k(x, y) = \left(
        1 + \sqrt{3}\|x - y\|
        \right)\exp\left(-\sqrt{3}\|x - y\|\right); $$

* `Matern52()`, the Matern–5/2 kernel:

    $$ k(x, y) = \left(
        1 + \sqrt{5}\|x - y\| + \frac{5}{3} \|x - y\|^2
       \right)\exp\left(-\sqrt{3}\|x - y\|\right); $$

* `Delta()`, the Kronecker delta kernel:

    $$ k(x, y) = \begin{cases}
        1 & \text{if } x = y, \\
        0 & \text{otherwise};
       \end{cases} $$
       
* `DecayingKernel(alpha, beta)`:

    $$ k(x, y) = \frac{\|\beta\|^\alpha}{\|x + y + \beta\|^\alpha}; $$
    
* `LogKernel()`:

    $$ k(x, y) = \frac{\log(1 + \|x - y\|)}{\|x - y\|}; $$

* `TensorProductKernel(f)`:

    $$ k(x, y) = f(x)f(y). $$

    Adding or multiplying a `FunctionType` `f` to or with a kernel will 
    automatically translate `f` to `TensorProductKernel(f)`. For example,
    `f * k` will translate to `TensorProductKernel(f) * k`, and `f + k` will 
    translate to `TensorProductKernel(f) + k`.


#### Compositional Design

* Add and subtract _means and kernels_.

    Example:
    
    ```python
    >>> EQ() + Exp()
    EQ() + Exp()

    >>> EQ() + EQ()
    2 * EQ()

    >>> EQ() + 1
    EQ() + 1

    >>> EQ() + 0
    EQ()

    >>> EQ() - Exp()
    EQ() - Exp()

    >>> EQ() - EQ()
    0
    ```

* Multiply _means and kernels_.
    
    Example:

    ```python
    >>> EQ() * Exp()
    EQ() * Exp()

    >>> 2 * EQ()
    2 * EQ()

    >>> 0 * EQ()
    0
    ```

* Shift _means and kernels_:

    Definition:
    
    ```python
    k.shift(c)(x, y) == k(x - c, y - c)

    k.shift(c1, c2)(x, y) == k(x - c1, y - c2)
    ```
    
    Example:
    
    ```python
    >>> Linear().shift(1)
    Linear() shift 1

    >>> EQ().shift(1, 2)
    EQ() shift (1, 2)
    ```

* Stretch _means and kernels_.

    Definition:
    
    ```python
    k.stretch(c)(x, y) == k(x / c, y / c)

    k.stretch(c1, c2)(x, y) == k(x / c1, y / c2)
    ```
  
    Example:    
    
    ```python
    >>> EQ().stretch(2)
    EQ() > 2

    >>> EQ().stretch(2, 3)
    EQ() > (2, 3)
    ```
    
    The `>` operator is implemented to provide a shorthand for stretching:
    
    ```python
    >>> EQ() > 2
    EQ() > 2
    ```

* Select particular input dimensions of _means and kernels_.

    Definition:

    ```python
    k.select([0])(x, y) == k(x[:, 0], y[:, 0])
  
    k.select([0, 1])(x, y) == k(x[:, [0, 1]], y[:, [0, 1]])

    k.select([0], [1])(x, y) == k(x[:, 0], y[:, 1])

    k.select(None, [1])(x, y) == k(x, y[:, 1])
    ```

    Example:

    ```python
    >>> EQ().select([0])
    EQ() : [0]
  
    >>> EQ().select([0, 1])
    EQ() : [0, 1]

    >>> EQ().select([0], [1])
    EQ() : ([0], [1])

    >>> EQ().select(None, [1])
    EQ() : (None, [1])
    ```

* Transform the inputs of _means and kernels_.

    Definition:

    ```python
    k.transform(f)(x, y) == k(f(x), f(y))

    k.transform(f1, f2)(x, y) == k(f1(x), f2(y))

    k.transform(None, f)(x, y) == k(x, f(y))
    ```
        
    Example:
        
    ```python
    >>> EQ().transform(f)
    EQ() transform f

    >>> EQ().transform(f1, f2)
    EQ() transform (f1, f2)

    >>> EQ().transform(None, f)
    EQ() transform (None, f)
    ```

* Numerically, but efficiently, take derivatives of _means and kernels_.
    This currently only works in TensorFlow.

    Definition:

    ```python
    k.diff(0)(x, y) == d/d(x[:, 0]) d/d(y[:, 0]) k(x, y)

    k.diff(0, 1)(x, y) == d/d(x[:, 0]) d/d(y[:, 1]) k(x, y)

    k.diff(None, 1)(x, y) == d/d(y[:, 1]) k(x, y)
    ```
        
    Example:

    ```python
    >>> EQ().diff(0)
    d(0) EQ()

    >>> EQ().diff(0, 1)
    d(0, 1) EQ()

    >>> EQ().diff(None, 1)
    d(None, 1) EQ()
    ```

*
    Make _kernels_ periodic.
    This is _not_ implemented for means.

    Definition:

    ```python
    k.periodic(2 pi / w)(x, y) == k((sin(w * x), cos(w * x)), (sin(w * y), cos(w * y)))
    ```

    Example:
     
    ```python
    >>> EQ().periodic(1)
    EQ() per 1
    ```

* 
    Reverse the arguments of _kernels_.
    This does _not_ apply to means.

    Definition:

    ```python
    reversed(k)(x, y) == k(y, x)
    ```

    Example:

    ```python
    >>> reversed(Linear())
    Reversed(Linear())
    ```
    
* Extract terms and factors from sums and products respectively of _means and 
kernels_.
    
    Example:
    
    ```python
    >>> (EQ() + RQ(0.1) + Linear()).term(1)
    RQ(0.1)

    >>> (2 * EQ() * Linear).factor(0)
    2
    ```
    
    Kernels and means "wrapping" others can be "unwrapped" by indexing `k[0]`
     or `m[0]`.
     
    Example:
    
    ```python
    >>> reversed(Linear())
    Reversed(Linear())
  
    >>> reversed(Linear())[0]
    Linear()

    >>> EQ().periodic(1)
    EQ() per 1

    >>> EQ().periodic(1)[0]
    EQ()
    ```

#### Displaying Means and Kernels

Kernels and means have a `display` method.
The `display` method accepts a callable formatter that will be applied before any value
is printed.
This comes in handy when pretty printing kernels.

Example:

```python
>>> print((2.12345 * EQ()).display(lambda x: f"{x:.2f}"))
2.12 * EQ(), 0
```

#### Properties of Means and Kernels

*
    Means and kernels can be equated to check for equality.
    This will attempt basic algebraic manipulations.
    If the means and kernels are not equal _or_ equality cannot be proved, `False` is
    returned.
    
    Example of equating kernels:

    ```python
    >>>  2 * EQ() == EQ() + EQ()
    True

    >>> EQ() + Exp() == Exp() + EQ()
    True

    >>> 2 * Exp() == EQ() + Exp()
    False

    >>> EQ() + Exp() + Linear()  == Linear() + Exp() + EQ()  # Too hard: cannot prove equality!
    False
    ```


*
    The stationarity of a kernel `k` can always be determined by querying
    `k.stationary`.

    Example of querying the stationarity:

    ```python
    >>> EQ().stationary
    True

    >>> (EQ() + Linear()).stationary
    False
    ```


## Examples

The examples make use of [Varz](https://github.com/wesselb/varz) and some
utility from [WBML](https://github.com/wesselb/wbml).

{examples}
