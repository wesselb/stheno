# [Stheno](https://github.com/wesselb/stheno)

[![Build](https://travis-ci.org/wesselb/stheno.svg?branch=master)](https://travis-ci.org/wesselb/stheno)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/stheno/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/stheno?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/stheno)

Stheno is an implementation of Gaussian process modelling in Python. See 
also [Stheno.jl](https://github.com/willtebbutt/Stheno.jl).

_Note:_ Stheno requires Python 3.5+ and TensorFlow 2 if TensorFlow is used.

* [Nonlinear Regression in 20 Seconds](#nonlinear-regression-in-20-seconds)
* [Installation](#installation)
* [Manual](#manual)
    - [Kernel and Mean Design](#kernel-and-mean-design)
        * [Available Kernels](#available-kernels)
        * [Available Means](#available-means)
        * [Compositional Design](#compositional-design)
        * [Displaying Kernels and Means](#displaying-kernels-and-mean)
        * [Properties of Kernels](#properties-of-kernels)
    - [Model Design](#model-design)
        * [Compositional Design](#compositional-design)
        * [Displaying GPs](#displaying-gps)
        * [Properties of GPs](#properties-of-gps)
        * [Naming GPs](#naming-gps)
    - [Finite-Dimensional Distributions, Inference, and Sampling](#finite-dimensional-distributions-inference-and-sampling)
    - [Inducing Points](#inducing-points)
    - [AutoGrad, TensorFlow, or PyTorch?](#autograd-tensorflow-or-pytorch)
    - [Undiscussed Features](#undiscussed-features)
* [Examples](#examples)
{examples_toc}

## Nonlinear Regression in 20 Seconds

```python
>>> import numpy as np

>>> from stheno import GP, EQ

>>> x = np.linspace(0, 2, 10)    # Points to predict at

>>> y = x ** 2                   # Observations

>>> (GP(EQ()) | (x, y))(3).mean  # Go GP!
array([[8.48258669]])
```

Moar?! Then read on!

## Installation

Before installing the package, please ensure that `gcc` and `gfortran` are 
available.
On OS X, these are both installed with `brew install gcc`;
users of Anaconda may want to instead consider `conda install gcc`.
On Linux, `gcc` is most likely already available, and `gfortran` can be
installed with `apt-get install gfortran`.
Then simply

```
pip install stheno
```

## Manual

Note: [here](https://wesselb.github.io/stheno) is a nicely rendered and more
readable version of the docs.

### Kernel and Mean Design

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

Finally, mean functions output a _rank 2 column vector_.

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

* `TensorProductKernel(f)`:

    $$ k(x, y) = f(x)f(y). $$

    Adding or multiplying a `FunctionType` `f` to or with a kernel will 
    automatically translate `f` to `TensorProductKernel(f)`. For example,
    `f * k` will translate to `TensorProductKernel(f) * k`, and `f + k` will 
    translate to `TensorProductKernel(f) + k`.


#### Available Means

Constants function as constant means. Besides that, the following means are 
available:

* `TensorProductMean(f)`:

    $$ m(x) = f(x). $$

    Adding or multiplying a `FunctionType` `f` to or with a mean will 
    automatically translate `f` to `TensorProductMean(f)`. For example,
    `f * m` will translate to `TensorProductMean(f) * m`, and `f + m` will 
    translate to `TensorProductMean(f) + m`.

#### Compositional Design

* Add and subtract _kernels and means_.

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

* Multiply _kernels and means_.
    
    Example:

    ```python
    >>> EQ() * Exp()
    EQ() * Exp()

    >>> 2 * EQ()
    2 * EQ()

    >>> 0 * EQ()
    0
    ```

* Shift _kernels and means_:

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

* Stretch _kernels and means_.

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

* Select particular input dimensions of _kernels and means_.

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

* Transform the inputs of _kernels and means_.

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

* Numerically, but efficiently, take derivatives of _kernels and means_.
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

* Make _kernels_ periodic, but _not means_.

    Definition:

    ```python
    k.periodic(2 pi / w)(x, y) == k((sin(w * x), cos(w * x)), (sin(w * y), cos(w * y)))
    ```

    Example:
     
    ```python
    >>> EQ().periodic(1)
    EQ() per 1
    ```

* Reverse the arguments of _kernels_, but _not means_.

    Definition:

    ```python
    reversed(k)(x, y) == k(y, x)
    ```

    Example:

    ```python
    >>> reversed(Linear())
    Reversed(Linear())
    ```
    
* Extract terms and factors from sums and products respectively of _kernels and 
means_.
    
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

#### Displaying Kernels and Means

Kernels and means have a `display` method.
The `display` method accepts a callable formatter that will be applied before any value is printed.
This comes in handy when pretty printing kernels, or when kernels contain TensorFlow objects.

Example:

```python
>>> print((2.12345 * EQ()).display(lambda x: '{:.2f}'.format(x)))
2.12 * EQ(), 0

>>> tf.constant(1) * EQ()
Tensor("Const_1:0", shape=(), dtype=int32) * EQ()

>>> print((tf.constant(2) * EQ()).display(tf.Session().run))
2 * EQ()
```

#### Properties of Kernels

The stationarity of a kernel `k` can always be determined by querying
`k.stationary`. In many cases, the variance `k.var`, length scale
`k.length_scale`, and period `k.period` can also be determined.

Example of querying the stationarity:

```python
>>> EQ().stationary
True

>>> (EQ() + Linear()).stationary
False
```

Example of querying the variance:

```python
>>> EQ().var
1

>>> (2 * EQ()).var
2
```

Example of querying the length scale:

```python
>>> EQ().length_scale
1

>>> (EQ() + EQ().stretch(2)).length_scale
1.5
```

Example of querying the period:

```python
>>> EQ().periodic(1).period
1

>>> EQ().periodic(1).stretch(2).period
2
```

### Model Design

The basic building block of a model is a `GP(kernel, mean=0, graph=model)`, 
which necessarily takes in a kernel, and optionally a mean and a _graph_.
GPs can be combined into new GPs, and the graph is the thing that keeps 
track of all of these objects.
If the graph is left unspecified, new GPs are appended to a provided default 
graph `model`, which is exported by Stheno:

```python
from stheno import model
```

Here's an example model:

```python
>>> f1 = GP(EQ(), lambda x: x ** 2)

>>> f1
GP(EQ(), <lambda>)

>>> f2 = GP(Linear())

>>> f_sum = f1 + f2

>>> f_sum
GP(EQ() + Linear(), <lambda>)
```


#### Compositional Design

* Add and subtract GPs and other objects.

    Example:
    
    ```python
    >>> GP(EQ()) + GP(Exp())
    GP(EQ() + Exp(), 0)

    >>> GP(EQ()) + GP(EQ())
    GP(2 * EQ(), 0)
  
    >>> GP(EQ()) + 1
    GP(EQ(), 1)
  
    >>> GP(EQ()) + 0
    GP(EQ(), 0)
  
    >>> GP(EQ()) + (lambda x: x ** 2)
    GP(EQ(), <lambda>)

    >>> GP(EQ(), 2) - GP(EQ(), 1)
    GP(2 * EQ(), 1)
    ```
    
* Multiply GPs by other objects.

    Example:
    
    ```python
    >>> 2 * GP(EQ())
    GP(2 * EQ(), 0)
  
    >>> 0 * GP(EQ())
    GP(0, 0)

    >>> (lambda x: x) * GP(EQ())
    GP(<lambda> * EQ(), 0)
    ```
    
* Shift GPs.

    Example:
    
    ```python
    >>> GP(EQ()).shift(1)
    GP(EQ() shift 1, 0) 
    ```
    
* Stretch GPs.

    Example:
    
    ```python
    >>> GP(EQ()).stretch(2)
    GP(EQ() > 2, 0)
    ```
    
    The `>` operator is implemented to provide a shorthand for stretching:
    
    ```python
    >>> GP(EQ()) > 2
    GP(EQ() > 2, 0)
    ```
    
* Select particular input dimensions.

    Example:
    
    ```python
    >>> GP(EQ()).select(1, 3)
    GP(EQ() : [1, 3], 0)
    ```
    
    Indexing is implemented to provide a a shorthand for selecting input 
    dimensions:
    
    ```python
    >>> GP(EQ())[1, 3]
    GP(EQ() : [1, 3], 0) 
    ```
    
* Transform the input.

    Example:
    
    ```python
    >>> GP(EQ()).transform(f)
    GP(EQ() transform f, 0)
    ```
    
* Numerically take the derivative of a GP.
    The argument specifies which dimension to take the derivative with respect
    to.
    
    Example:
    
    ```python
    >>> GP(EQ()).diff(1)
    GP(d(1) EQ(), 0)
    ```
    
* Construct a finite difference estimate of the derivative of a GP.
    See `stheno.graph.Graph.diff_approx` for a description of the arguments.
    
    Example:
    
    ```python
    >>> GP(EQ()).diff_approx(deriv=1, order=2)
    GP(50000000.0 * (0.5 * EQ() + 0.5 * ((-0.5 * (EQ() shift (0.0001414213562373095, 0))) shift (0, -0.0001414213562373095)) + 0.5 * ((-0.5 * (EQ() shift (0, 0.0001414213562373095))) shift (-0.0001414213562373095, 0))), 0)
    ```
    
* Construct the Cartesian product of a collection of GPs.

    Example:
    
    ```python
    >>> model = Graph()

    >>> f1, f2 = GP(EQ(), graph=model), GP(EQ(), graph=model)

    >>> model.cross(f1, f2)
    GP(MultiOutputKernel(EQ(), EQ()), MultiOutputMean(0, 0))
    ```

#### Displaying GPs

Like kernels and means, GPs have a `display` method that accepts a formatter.

Example:

```python
>>> print(GP(2.12345 * EQ()).display(lambda x: '{:.2f}'.format(x)))
GP(2.12 * EQ(), 0)
```

#### Properties of GPs

Properties of kernels can be queried on GPs directly.

Example:

```python
>>> GP(EQ()).stationary
True

>>> GP(RQ(1e-1)).length_scale
1
```

### Naming GPs

It is possible to give a name to GPs.
Names must be strings.
A graph then behaves like a two-way dictionary between GPs and their names.

Example:

```python
>>> p = GP(EQ(), name='prior')

>>> p.name
'prior'

>>> p.name = 'alternative_prior'

>>> model['alternative_prior']
GP(EQ(), 0)

>>> model[p]
'alternative_prior'
```

### Finite-Dimensional Distributions, Inference, and Sampling


Simply call a GP to construct its finite-dimensional distribution:

```python
>>> type(f(x))
stheno.random.Normal

>>> f(x).mean
array([[0.],
       [0.],
       [0.]])

>>> f(x).var
array([[1.        , 0.8824969 , 0.60653066],
       [0.8824969 , 1.        , 0.8824969 ],
       [0.60653066, 0.8824969 , 1.        ]])
       
>>> f(x).sample(1)
array([[-0.47676132],
       [-0.51696144],
       [-0.77643117]])
       
>>> y1 = f(x).sample(1)

>>> f(x).logpdf(y1)
-1.348196150807441

>>> y2 = f(x).sample(2)

>>> f(x).logpdf(y2)
 array([-1.00581476, -1.67625465])
```

If you wish to compute the evidence of multiple observations, 
then `Graph.logpdf` can be used.

Definition:

```python
Graph.logpdf(f(x), y)

Graph.logpdf((f1(x1), y1), (f2(x2), y2), ...)
```

Furthermore, use `f(x).marginals()` to efficiently compute the means and 
the marginal lower and upper 95% central credible region bounds:

```python
>>> f(x).marginals()
(array([0., 0., 0.]), array([-2., -2., -2.]), array([2., 2., 2.]))
```

To condition on observations, use `Graph.condition` or `GP.condition`.
Syntax is much like the math:
compare `f1_posterior = f1 | (f2(x), y)` with $f_1 \,|\, f_2(x) = y$.

Definition, where `f*` and `g*` are `GP`s:

```python
f_posterior = f | (x, y)

f_posterior = f | (g1(x), y)

f_posterior = f | ((g1(x1), y1), (g2(x2), y2), ...)

f1_posterior, f2_posterior, ... = (f1, f2, ...) | Obs(g(x), y)

f1_posterior, f2_posterior, ... = (f1, f2, ...) | Obs((g1(x1), y1), (g2(x2), y2), ...)

```

Finally, `Graph.sample` can be used to get samples from multiple processes 
jointly:

```python
>>> model.sample(f(x), (2 * f)(x))
[array([[-0.35226314],
        [-0.15521219],
        [ 0.0752406 ]]),
 array([[-0.70452827],
        [-0.31042226],
        [ 0.15048168]])]
```

### Inducing Points

Stheno supports sparse approximations of posterior distributions. To construct
a sparse approximation, use `SparseObs` instead of `Obs`.

Definition:

```python
obs = SparseObs(u(z),  # Locations of inducing points.
                e,     # Independent, additive noise process.
                f(x),  # Locations of observations _without_ the noise 
                       #   process added.
                y)     # Observations.
                
obs = SparseObs(u(z), e, f(x), y)

obs = SparseObs(u(z), (e1, f1(x1), y1), (e2, f2(x2), y2), ...)

obs = SparseObs((u1(z1), u2(z2), ...), e, f(x), y)

obs = SparseObs(u(z), (e1, f1(x1), y1), (e2, f2(x2), y2), ...)

obs = SparseObs((u1(z1), u2(z2), ...), (e1, f1(x1), y1), (e2, f2(x2), y2), ...)
```

`SparseObs` will also compute the value of the ELBO in `obs.elbo`, which can be
optimised to select hyperparameters and locations of the inducing points.

### AutoGrad, TensorFlow, or PyTorch?

Your choice!

```python
from stheno.autograd import GP, EQ
```

```python
from stheno.tensorflow import GP, EQ
```

```python
from stheno.torch import GP, EQ
```

### Undiscussed Features

* `stheno.mokernel` and `stheno.momean` offer multi-output kernels and means.

    Example:

    ```python
    >>> model = Graph()

    >>> f1, f2 = GP(EQ(), graph=model), GP(EQ(), graph=model)

    >>> f = model.cross(f1, f2)

    >>> f
    GP(MultiOutputKernel(EQ(), EQ()), MultiOutputMean(0, 0))

    >>> f(0).sample()
    array([[ 1.1725799 ],
           [-1.15642448]])
    ```

* `stheno.eis` offers kernels on an extended input space that allows one to 
design kernels in an alternative, flexible way.

    Example:

    ```python
    >>> p = GP(NoisyKernel(EQ(), Delta()))

    >>> prediction = p.condition(Observed(x), y)(Latent(x)).marginals()
    ```
    
* `stheno.normal` offers an efficient implementation `Normal` of the normal 
distribution, and a convenience constructor `Normal1D` for 1-dimensional normal
distributions.

* `stheno.matrix` offers structured representations of matrices and efficient
operations thereon.

* Approximate multiplication between GPs is implemented. This is an 
experimental feature.

    Example:
    
    ```python
    >>> GP(EQ(), 1) * GP(EQ(), 1)
    GP(<lambda> * EQ() + <lambda> * EQ() + EQ() * EQ(), <lambda> + <lambda> + -1 * 1)
    ```

## Examples

The examples make use of [Varz](https://github.com/wesselb/varz) and some
utility from [WBML](https://github.com/wesselb/wbml).

{examples}
