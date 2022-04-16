# [Stheno](https://github.com/wesselb/stheno)

[![CI](https://github.com/wesselb/stheno/workflows/CI/badge.svg?branch=master)](https://github.com/wesselb/stheno/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/stheno/badge.svg?branch=master)](https://coveralls.io/github/wesselb/stheno?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/stheno)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Stheno is an implementation of Gaussian process modelling in Python. See 
also [Stheno.jl](https://github.com/willtebbutt/Stheno.jl).

[Check out our post about linear models with Stheno and JAX.](https://wesselb.github.io/2021/01/19/linear-models-with-stheno-and-jax.html)

Contents:

* [Nonlinear Regression in 20 Seconds](#nonlinear-regression-in-20-seconds)
* [Installation](#installation)
* [Manual](#manual)
    - [AutoGrad, TensorFlow, PyTorch, or JAX? Your Choice!](#autograd-tensorflow-pytorch-or-jax-your-choice)
    - [Model Design](#model-design)
    - [Finite-Dimensional Distributions](#finite-dimensional-distributions)
    - [Prior and Posterior Measures](#prior-and-posterior-measures)
    - [Inducing Points](#inducing-points)
    - [Kernels and Means](#kernels-and-means)
    - [Batched Computation](#batched-computation)
    - [Important Remarks](#important-remarks)
* [Examples](#examples)
{examples_toc}

## Nonlinear Regression in 20 Seconds

```python
>>> import numpy as np

>>> from stheno import GP, EQ

>>> x = np.linspace(0, 2, 10)           # Some points to predict at

>>> y = x ** 2                          # Some observations

>>> f = GP(EQ())                        # Construct Gaussian process.

>>> f_post = f | (f(x), y)              # Compute the posterior.

>>> pred = f_post(np.array([1, 2, 3]))  # Predict!

>>> pred.mean
<dense matrix: shape=3x1, dtype=float64
 mat=[[1.   ]
      [4.   ]
      [8.483]]>

>>> pred.var
<dense matrix: shape=3x3, dtype=float64
 mat=[[ 8.032e-13  7.772e-16 -4.577e-09]
      [ 7.772e-16  9.999e-13  2.773e-10]
      [-4.577e-09  2.773e-10  3.313e-03]]>
```

[These custom matrix types are there to accelerate the underlying linear algebra.](#important-remarks)
To get vanilla NumPy/AutoGrad/TensorFlow/PyTorch/JAX arrays, use `B.dense`:

```python
>>> from lab import B

>>> B.dense(pred.mean)
array([[1.00000068],
       [3.99999999],
       [8.4825932 ]])

>>> B.dense(pred.var)
array([[ 8.03246358e-13,  7.77156117e-16, -4.57690943e-09],
       [ 7.77156117e-16,  9.99866856e-13,  2.77333267e-10],
       [-4.57690943e-09,  2.77333267e-10,  3.31283378e-03]])
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

### AutoGrad, TensorFlow, PyTorch, or JAX? Your Choice!

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
in [a _mean_, a _kernel_](#kernels-and-means), and a _measure_.
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

To avoid setting the keyword `measure` for every `GP` that you create, you can enter
a measure as a context:

```python
>>> with Measure() as prior:
        f1 = GP(lambda x: x ** 2, EQ())
        f2 = GP(Linear())
        f_sum = f1 + f2

>>> prior == f1.measure == f2.measure == f_sum.measure
True
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
    
* Multiply GPs and other objects.

    *Warning:*
    The product of two GPs it *not* a Gaussian process.
    Stheno approximates the resulting process by moment matching.

    Example:
    
    ```python
    >>> GP(1, EQ(), measure=prior) * GP(1, Exp(), measure=prior)
    GP(<lambda> + <lambda> + -1 * 1, <lambda> * Exp() + <lambda> * EQ() + EQ() * Exp())
  
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
    
* Select particular input dimensions.

    Example:
    
    ```python
    >>> GP(EQ()).select(1, 3)
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
    See `Measure.diff_approx` for a description of the arguments.
    
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

GPs have a `display` method that accepts a formatter.

Example:

```python
>>> print(GP(2.12345 * EQ()).display(lambda x: f"{x:.2f}"))
GP(2.12 * EQ(), 0)
```

#### Properties of GPs

[Properties of kernels](https://github.com/wesselb/mlkernels#properties-of-kernels-and-means)
can be queried on GPs directly.

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

>>> p = GP(EQ(), name="name", measure=prior)

>>> p.name
'name'

>>> p.name = "alternative_name"

>>> prior["alternative_name"]
GP(0, EQ())

>>> prior[p]
'alternative_name'
```

### Finite-Dimensional Distributions

Simply call a GP to construct a finite-dimensional distribution at some inputs.
You can give a second argument, which specifies the variance of additional additive
noise.
After constructing a finite-dimensional distribution, you can compute the mean,
the variance, sample, or compute a logpdf.

Definition, where `f` is a `GP`:

```python
f(x)         # No additional noise

f(x, noise)  # Additional noise with variance `noise`
```

Things you can do with a finite-dimensional distribution:

* 
    Use `f(x).mean` to compute the mean.
    
* 
    Use `f(x).var` to compute the variance.
 
* 
    Use `f(x).mean_var` to compute simultaneously compute the mean and variance.
    This can be substantially more efficient than calling first `f(x).mean` and then
    `f(x).var`.

* 
    Use `Normal.sample` to sample.

    Definition:
  
    ```python
    f(x).sample()                # Produce one sample.
  
    f(x).sample(n)               # Produce `n` samples.
  
    f(x).sample(noise=noise)     # Produce one samples with additional noise variance `noise`.
  
    f(x).sample(n, noise=noise)  # Produce `n` samples with additional noise variance `noise`.
    ```
  
* 
    Use `f(x).logpdf(y)` to compute the logpdf of some data `y`.
    
* 
    Use `means, variances = f(x).marginals()` to efficiently compute the marginal means
    and marginal variances.
    
    Example:

    ```python
    >>> f(x).marginals()
    (array([0., 0., 0.]), np.array([1., 1., 1.]))
    ```
  
* 
    Use `means, lowers, uppers = f(x).marginal_credible_bounds()` to efficiently compute
    the means and the marginal lower and upper 95% central credible region bounds.

    Example:

    ```python
    >>> f(x).marginal_credible_bounds()
    (array([0., 0., 0.]), array([-1.96, -1.96, -1.96]), array([1.96, 1.96, 1.96]))
    ```
  
* 
    Use `Measure.logpdf` to compute the joint logpdf of multiple observations.

    Definition, where `prior = Measure()`:

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

Example:

```python
>>> prior = Measure()

>>> f = GP(EQ(), measure=prior)

>>> x = np.array([0., 1., 2.])

>>> f(x)       # FDD without noise.
<FDD:
 process=GP(0, EQ()),
 input=array([0., 1., 2.]),
 noise=<zero matrix: shape=3x3, dtype=float64>

>>> f(x, 0.1)  # FDD with noise.
<FDD:
 process=GP(0, EQ()),
 input=array([0., 1., 2.]),
 noise=<diagonal matrix: shape=3x3, dtype=float64
        diag=[0.1 0.1 0.1]>>

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

### Prior and Posterior Measures

Conditioning a _prior_ measure on observations gives a _posterior_ measure.
To condition a measure on observations, use `Measure.__or__`.

Definition, where `prior = Measure()` and `f*` are `GP`s:

```python
post = prior | (f(x, [noise]), y)

post = prior | ((f1(x1, [noise1]), y1), (f2(x2, [noise2]), y2), ...)
```

You can then obtain a posterior process with `post(f)` and a finite-dimensional
distribution under the posterior with `post(f(x))`.
Alternatively, the posterior of a process `f` can be obtained by conditioning `f`
directly.

Definition, where and `f*` are `GP`s:

```python
f_post = f | (f(x, [noise]), y)

f_post = f | ((f1(x1, [noise1]), y1), (f2(x2, [noise2]), y2), ...)
```

Let's consider an example.
First, build a model and sample some values.

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
<FDD:
 process=GP(PosteriorMean(), PosteriorKernel()),
 input=array([0., 1., 2.]),
 noise=<zero matrix: shape=3x3, dtype=float64>>

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

We can also obtain the posterior by conditioning `f` directly:

```python
>>> f_post = f | (f(x), y)

>>> f_post
GP(PosteriorMean(), PosteriorKernel())

>>> f_post.mean(x)
<dense matrix: shape=3x1, dtype=float64
 mat=[[ 0.412]
      [-0.811]
      [-0.933]]>

>>> f_post.kernel(x)
<dense matrix: shape=3x3, dtype=float64
 mat=[[1.e-12 0.e+00 0.e+00]
      [0.e+00 1.e-12 0.e+00]
      [0.e+00 0.e+00 1.e-12]]>

>>> f_post(x)
<FDD:
 process=GP(PosteriorMean(), PosteriorKernel()),
 input=array([0., 1., 2.]),
 noise=<zero matrix: shape=3x3, dtype=float64>>

>>> f_post(x).mean
<dense matrix: shape=3x1, dtype=float64
 mat=[[ 0.412]
      [-0.811]
      [-0.933]]>

>>> f_post(x).var
<dense matrix: shape=3x3, dtype=float64
 mat=[[1.e-12 0.e+00 0.e+00]
      [0.e+00 1.e-12 0.e+00]
      [0.e+00 0.e+00 1.e-12]]>
```

We can further extend our model by building on the posterior.

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

Stheno supports pseudo-point approximations of posterior distributions with
various approximation methods:

1. The Variational Free Energy (VFE;
    [Titsias, 2009](http://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf))
    approximation.
    To use the VFE approximation, use `PseudoObs`.

2. The Fully Independent Training Conditional (FITC;
    [Snelson & Ghahramani, 2006](http://www.gatsby.ucl.ac.uk/~snelson/SPGP_up.pdf))
    approximation. 
    To use the FITC approximation, use `PseudoObsFITC`.
 
3. The Deterministic Training Conditional (DTC;
   [Csato & Opper, 2002](https://direct.mit.edu/neco/article/14/3/641/6594/Sparse-On-Line-Gaussian-Processes);
   [Seeger et al., 2003](http://proceedings.mlr.press/r4/seeger03a/seeger03a.pdf))
   approximation.
   To use the DTC approximation, use `PseudoObsDTC`.

The VFE approximation (`PseudoObs`) is the approximation recommended to use.
The following definitions and examples will use the VFE approximation with `PseudoObs`,
but every instance of `PseudoObs` can be swapped out for `PseudoObsFITC` or 
`PseudoObsDTC`.

Definition:

```python
obs = PseudoObs(
    u(z),               # FDD of inducing points
    (f(x, [noise]), y)  # Observed data
)
                
obs = PseudoObs(u(z), f(x, [noise]), y)

obs = PseudoObs(u(z), (f1(x1, [noise1]), y1), (f2(x2, [noise2]), y2), ...)

obs = PseudoObs((u1(z1), u2(z2), ...), f(x, [noise]), y)

obs = PseudoObs((u1(z1), u2(z2), ...), (f1(x1, [noise1]), y1), (f2(x2, [noise2]), y2), ...)
```

The approximate posterior measure can be constructed with `prior | obs`
where `prior = Measure()` is the measure of your model.
To quantify the quality of the approximation, you can compute the ELBO with 
`obs.elbo(prior)`.

Let's consider an example.
First, build a model and sample some noisy observations.

```python
>>> prior = Measure()

>>> f = GP(EQ(), measure=prior)

>>> x_obs = np.linspace(0, 10, 2000)

>>> y_obs = f(x_obs, 1).sample()
```

Ouch, computing the logpdf is quite slow:

```python
>>> %timeit f(x_obs, 1).logpdf(y_obs)
219 ms ± 35.7 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

Let's try to use inducing points to speed this up.

```python
>>> x_ind = np.linspace(0, 10, 100)

>>> u = f(x_ind)   # FDD of inducing points.

>>> %timeit PseudoObs(u, f(x_obs, 1), y_obs).elbo(prior)
9.8 ms ± 181 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

Much better.
And the approximation is good:

```python
>>> PseudoObs(u, f(x_obs, 1), y_obs).elbo(prior) - f(x_obs, 1).logpdf(y_obs)
-3.537934389896691e-10
```

We finally construct the approximate posterior measure:

```python
>>> post_approx = prior | PseudoObs(u, f(x_obs, 1), y_obs)

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


### Kernels and Means

See [MLKernels](https://github.com/wesselb/mlkernels).


### Batched Computation

Stheno supports batched computation.
See [MLKernels](https://github.com/wesselb/mlkernels/#usage) for a description of how
means and kernels work with batched computation.

Example:

```python
>>> f = GP(EQ())

>>> x = np.random.randn(16, 100, 1)

>>> y = f(x, 1).sample()

>>> logpdf = f(x, 1).logpdf(y)

>>> y.shape
(16, 100, 1)

>>> f(x, 1).logpdf(y).shape
(16,)
```


### Important Remarks

Stheno uses [LAB](https://github.com/wesselb/lab) to provide an implementation that is
backend agnostic.
Moreover, Stheno uses [an extension of LAB](https://github.com/wesselb/matrix) to
accelerate linear algebra with structured linear algebra primitives.
You will encounter these primitives:

```python
>>> k = 2 * Delta()

>>> x = np.linspace(0, 5, 10)

>>> k(x)
<diagonal matrix: shape=10x10, dtype=float64
 diag=[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]>
```

If you're using [LAB](https://github.com/wesselb/lab) to further process these matrices,
then there is absolutely no need to worry:
these structured matrix types know how to add, multiply, and do other linear algebra
operations.

```python
>>> import lab as B

>>> B.matmul(k(x), k(x))
<diagonal matrix: shape=10x10, dtype=float64
 diag=[4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]>
```

If you're not using [LAB](https://github.com/wesselb/lab), you can convert these
structured primitives to regular NumPy/TensorFlow/PyTorch/JAX arrays by calling
`B.dense` (`B` is from [LAB](https://github.com/wesselb/lab)):

```python
>>> import lab as B

>>> B.dense(k(x))
array([[2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 2., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 2., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 2., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 2., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 2., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 2., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 2., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 2., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 2.]])
```

Furthermore, before computing a Cholesky decomposition, Stheno always adds a minuscule
diagonal to prevent the Cholesky decomposition from failing due to positive
indefiniteness caused by numerical noise.
You can change the magnitude of this diagonal by changing `B.epsilon`:

```python
>>> import lab as B

>>> B.epsilon = 1e-12   # Default regularisation

>>> B.epsilon = 1e-8    # Strong regularisation
```


## Examples

The examples make use of [Varz](https://github.com/wesselb/varz) and some
utility from [WBML](https://github.com/wesselb/wbml).

{examples}
