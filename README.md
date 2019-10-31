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
    - [Simple Regression](#simple-regression)
    - [Decomposition of Prediction](#decomposition-of-prediction)
    - [Learn a Function, Incorporating Prior Knowledge About Its Form](#learn-a-function-incorporating-prior-knowledge-about-its-form)
    - [Multi-Output Regression](#multi-ouput-regression)
    - [Approximate Integration](#approximate-integration)
    - [Bayesian Linear Regression](#bayesian-linear-regression)
    - [GPAR](#gpar)
    - [A GP-RNN Model](#a-gp-rnn-model)
    - [Approximate Multiplication Between GPs](#approximate-multiplication-between-gps)
    - [Sparse Regression](#sparse-regression)
    - [Smoothing with Nonparametric Basis Functions](#smoothing-with-nonparametric-basis-functions)

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


### Simple Regression

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_example1_simple_regression.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import wbml.plot

from stheno import GP, EQ, Delta, model

# Define points to predict at.
x = np.linspace(0, 10, 100)
x_obs = np.linspace(0, 7, 20)

# Construct a prior.
f = GP(EQ().periodic(5.))  # Latent function.
e = GP(Delta())  # Noise.
y = f + .5 * e

# Sample a true, underlying function and observations.
f_true, y_obs = model.sample(f(x), y(x_obs))

# Now condition on the observations to make predictions.
mean, lower, upper = (f | (y(x_obs), y_obs))(x).marginals()

# Plot result.
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')

wbml.plot.tweak()
plt.savefig('readme_example1_simple_regression.png')
plt.show()
```

### Decomposition of Prediction

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_example2_decomposition.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import wbml.plot

from stheno import GP, model, EQ, RQ, Linear, Delta, Exp, Obs, B

B.epsilon = 1e-10

# Define points to predict at.
x = np.linspace(0, 10, 200)
x_obs = np.linspace(0, 7, 50)

# Construct a latent function consisting of four different components.
f_smooth = GP(EQ())
f_wiggly = GP(RQ(1e-1).stretch(.5))
f_periodic = GP(EQ().periodic(1.))
f_linear = GP(Linear())

f = f_smooth + f_wiggly + f_periodic + .2 * f_linear

# Let the observation noise consist of a bit of exponential noise.
e_indep = GP(Delta())
e_exp = GP(Exp())

e = e_indep + .3 * e_exp

# Sum the latent function and observation noise to get a model for the
# observations.
y = f + .5 * e

# Sample a true, underlying function and observations.
f_true_smooth, f_true_wiggly, f_true_periodic, f_true_linear, f_true, y_obs = \
    model.sample(f_smooth(x),
                 f_wiggly(x),
                 f_periodic(x),
                 f_linear(x),
                 f(x),
                 y(x_obs))

# Now condition on the observations and make predictions for the latent
# function and its various components.
f_smooth, f_wiggly, f_periodic, f_linear, f = \
    (f_smooth, f_wiggly, f_periodic, f_linear, f) | Obs(y(x_obs), y_obs)

pred_smooth = f_smooth(x).marginals()
pred_wiggly = f_wiggly(x).marginals()
pred_periodic = f_periodic(x).marginals()
pred_linear = f_linear(x).marginals()
pred_f = f(x).marginals()


# Plot results.
def plot_prediction(x, f, pred, x_obs=None, y_obs=None):
    plt.plot(x, f, label='True', c='tab:blue')
    if x_obs is not None:
        plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
    mean, lower, upper = pred
    plt.plot(x, mean, label='Prediction', c='tab:green')
    plt.plot(x, lower, ls='--', c='tab:green')
    plt.plot(x, upper, ls='--', c='tab:green')
    wbml.plot.tweak()


plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.title('Prediction')
plot_prediction(x, f_true, pred_f, x_obs, y_obs)

plt.subplot(3, 2, 3)
plt.title('Smooth Component')
plot_prediction(x, f_true_smooth, pred_smooth)

plt.subplot(3, 2, 4)
plt.title('Wiggly Component')
plot_prediction(x, f_true_wiggly, pred_wiggly)

plt.subplot(3, 2, 5)
plt.title('Periodic Component')
plot_prediction(x, f_true_periodic, pred_periodic)

plt.subplot(3, 2, 6)
plt.title('Linear Component')
plot_prediction(x, f_true_linear, pred_linear)

plt.savefig('readme_example2_decomposition.png')
plt.show()
```

### Learn a Function, Incorporating Prior Knowledge About Its Form

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_example3_parametric.png)

```python
import matplotlib.pyplot as plt
import tensorflow as tf
import wbml.out
import wbml.plot
from varz.tensorflow import Vars, minimise_l_bfgs_b

from stheno.tensorflow import B, Graph, GP, EQ, Delta

# Define points to predict at.
x = B.linspace(tf.float64, 0, 5, 100)
x_obs = B.linspace(tf.float64, 0, 3, 20)


def model(vs):
    g = Graph()

    # Random fluctuation:
    u = GP(vs.pos(.5, name='u/var') *
           EQ().stretch(vs.pos(0.5, name='u/scale')), graph=g)

    # Noise:
    e = GP(vs.pos(0.5, name='e/var') * Delta(), graph=g)

    # Construct model:
    alpha = vs.pos(1.2, name='alpha')
    f = u + (lambda x: x ** alpha)
    y = f + e

    return f, y


# Sample a true, underlying function and observations.
vs = Vars(tf.float64)
f_true = x ** 1.8 + B.sin(2 * B.pi * x)
f, y = model(vs)
y_obs = (y | (f(x), f_true))(x_obs).sample()


def objective(vs):
    f, y = model(vs)
    evidence = y(x_obs).logpdf(y_obs)
    return -evidence


# Learn hyperparameters.
minimise_l_bfgs_b(tf.function(objective, autograph=False), vs)
f, y = model(vs)

# Print the learned parameters.
wbml.out.kv('Alpha', vs['alpha'])
wbml.out.kv('Prior', y.display(wbml.out.format))

# Condition on the observations to make predictions.
mean, lower, upper = (f | (y(x_obs), y_obs))(x).marginals()

# Plot result.
plt.plot(x, B.squeeze(f_true), label='True', c='tab:blue')
plt.scatter(x_obs, B.squeeze(y_obs), label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
wbml.plot.tweak()

plt.savefig('readme_example3_parametric.png')
plt.show()
```

### Multi-Output Regression

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_example4_multi-output.png)

```python
import matplotlib.pyplot as plt
import numpy as np
from plum import Dispatcher, Referentiable, Self
import wbml.plot

from stheno import GP, EQ, Delta, model, Kernel, Obs


class VGP(Referentiable):
    """A vector-valued GP.

    Args:
        dim (int): Dimensionality.
        kernel (instance of :class:`stheno.kernel.Kernel`): Kernel.
    """
    dispatch = Dispatcher(in_class=Self)

    @dispatch(int, Kernel)
    def __init__(self, dim, kernel):
        self.ps = [GP(kernel) for _ in range(dim)]

    @dispatch([GP])
    def __init__(self, *ps):
        self.ps = ps

    @dispatch(Self)
    def __add__(self, other):
        return VGP(*[f + g for f, g in zip(self.ps, other.ps)])

    @dispatch(np.ndarray)
    def lmatmul(self, A):
        m, n = A.shape
        ps = [0 for _ in range(m)]
        for i in range(m):
            for j in range(n):
                ps[i] += A[i, j] * self.ps[j]
        return VGP(*ps)

    def sample(self, x):
        return model.sample(*(p(x) for p in self.ps))

    def __or__(self, obs):
        return VGP(*(p | obs for p in self.ps))

    def obs(self, x, ys):
        return Obs(*((p(x), y) for p, y in zip(self.ps, ys)))

    def marginals(self, x):
        return [p(x).marginals() for p in self.ps]


# Define points to predict at.
x = np.linspace(0, 10, 100)
x_obs = np.linspace(0, 10, 10)

# Model parameters:
m = 2
p = 4
H = np.random.randn(p, m)

# Construct latent functions
us = VGP(m, EQ())
fs = us.lmatmul(H)

# Construct noise.
e = VGP(p, 0.5 * Delta())

# Construct observation model.
ys = e + fs

# Sample a true, underlying function and observations.
fs_true = fs.sample(x)
ys_obs = (ys | fs.obs(x, fs_true)).sample(x_obs)

# Condition the model on the observations to make predictions.
preds = (fs | ys.obs(x_obs, ys_obs)).marginals(x)


# Plot results.
def plot_prediction(x, f, pred, x_obs=None, y_obs=None):
    plt.plot(x, f, label='True', c='tab:blue')
    if x_obs is not None:
        plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
    mean, lower, upper = pred
    plt.plot(x, mean, label='Prediction', c='tab:green')
    plt.plot(x, lower, ls='--', c='tab:green')
    plt.plot(x, upper, ls='--', c='tab:green')
    wbml.plot.tweak()


plt.figure(figsize=(10, 6))

for i in range(p):
    plt.subplot(int(p ** .5), int(p ** .5), i + 1)
    plt.title('Output {}'.format(i + 1))
    plot_prediction(x, fs_true[i], preds[i], x_obs, ys_obs[i])

plt.savefig('readme_example4_multi-output.png')
plt.show()
```

### Approximate Integration

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_example5_integration.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wbml.plot

from stheno.tensorflow import B, GP, EQ, Delta, Obs

# Define points to predict at.
x = B.linspace(tf.float64, 0, 10, 200)
x_obs = B.linspace(tf.float64, 0, 10, 10)

# Construct the model.
f = 0.7 * GP(EQ()).stretch(1.5)
e = 0.2 * GP(Delta())

# Construct derivatives.
df = f.diff()
ddf = df.diff()
dddf = ddf.diff() + e

# Fix the integration constants.
zero = tf.constant(0, dtype=tf.float64)
one = tf.constant(1, dtype=tf.float64)
f, df, ddf, dddf = (f, df, ddf, dddf) | Obs((f(zero), one),
                                            (df(zero), zero),
                                            (ddf(zero), -one))

# Sample observations.
y_obs = B.sin(x_obs) + 0.2 * B.randn(*x_obs.shape)

# Condition on the observations to make predictions.
f, df, ddf, dddf = (f, df, ddf, dddf) | Obs(dddf(x_obs), y_obs)

# And make predictions.
pred_iiif = f(x).marginals()
pred_iif = df(x).marginals()
pred_if = ddf(x).marginals()
pred_f = dddf(x).marginals()


# Plot result.
def plot_prediction(x, f, pred, x_obs=None, y_obs=None):
    plt.plot(x, f, label='True', c='tab:blue')
    if x_obs is not None:
        plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
    mean, lower, upper = pred
    plt.plot(x, mean, label='Prediction', c='tab:green')
    plt.plot(x, lower, ls='--', c='tab:green')
    plt.plot(x, upper, ls='--', c='tab:green')
    wbml.plot.tweak()


plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.title('Function')
plot_prediction(x, np.sin(x), pred_f, x_obs=x_obs, y_obs=y_obs)

plt.subplot(2, 2, 2)
plt.title('Integral of Function')
plot_prediction(x, -np.cos(x), pred_if)

plt.subplot(2, 2, 3)
plt.title('Second Integral of Function')
plot_prediction(x, -np.sin(x), pred_iif)

plt.subplot(2, 2, 4)
plt.title('Third Integral of Function')
plot_prediction(x, np.cos(x), pred_iiif)

plt.savefig('readme_example5_integration.png')
plt.show()
```

### Bayesian Linear Regression

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_example6_blr.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import wbml.out
import wbml.plot

from stheno import GP, Delta, model, Obs

# Define points to predict at.
x = np.linspace(0, 10, 200)
x_obs = np.linspace(0, 10, 10)

# Construct the model.
slope = GP(1)
intercept = GP(5)
f = slope * (lambda x: x) + intercept

e = 0.2 * GP(Delta())  # Noise model

y = f + e  # Observation model

# Sample a slope, intercept, underlying function, and observations.
true_slope, true_intercept, f_true, y_obs = \
    model.sample(slope(0), intercept(0), f(x), y(x_obs))

# Condition on the observations to make predictions.
slope, intercept, f = (slope, intercept, f) | Obs(y(x_obs), y_obs)
mean, lower, upper = f(x).marginals()

wbml.out.kv('True slope', true_slope[0, 0])
wbml.out.kv('Predicted slope', slope(0).mean[0, 0])
wbml.out.kv('True intercept', true_intercept[0, 0])
wbml.out.kv('Predicted intercept', intercept(0).mean[0, 0])

# Plot result.
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
wbml.plot.tweak()

plt.savefig('readme_example6_blr.png')
plt.show()
```

### GPAR

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_example7_gpar.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wbml.plot
from varz.tensorflow import Vars, minimise_l_bfgs_b

from stheno.tensorflow import B, Graph, GP, Delta, EQ

# Define points to predict at.
x = B.linspace(tf.float64, 0, 10, 200)
x_obs1 = B.linspace(tf.float64, 0, 10, 30)
inds2 = np.random.permutation(len(x_obs1))[:10]
x_obs2 = B.take(x_obs1, inds2)

# Construction functions to predict and observations.
f1_true = B.sin(x)
f2_true = B.sin(x) ** 2

y1_obs = B.sin(x_obs1) + 0.1 * B.randn(*x_obs1.shape)
y2_obs = B.sin(x_obs2) ** 2 + 0.1 * B.randn(*x_obs2.shape)


def model(vs):
    g = Graph()

    # Construct model for first layer:
    f1 = GP(vs.pos(1., name='f1/var') *
            EQ().stretch(vs.pos(1., name='f1/scale')), graph=g)
    e1 = GP(vs.pos(0.1, name='e1/var') * Delta(), graph=g)
    y1 = f1 + e1

    # Construct model for second layer:
    f2 = GP(vs.pos(1., name='f2/var') *
            EQ().stretch(vs.pos(np.array([1., .5]), name='f2/scale')), graph=g)
    e2 = GP(vs.pos(0.1, name='e2/var') * Delta(), graph=g)
    y2 = f2 + e2

    return f1, y1, f2, y2


def objective(vs):
    f1, y1, f2, y2 = model(vs)

    x1 = x_obs1
    x2 = B.stack(x_obs2, B.take(y1_obs, inds2), axis=1)
    evidence = y1(x1).logpdf(y1_obs) + y2(x2).logpdf(y2_obs)

    return -evidence


# Learn hyperparameters.
vs = Vars(tf.float64)
minimise_l_bfgs_b(objective, vs)
f1, y1, f2, y2 = model(vs)

# Condition to make predictions.
x1 = x_obs1
x2 = B.stack(x_obs2, B.take(y1_obs, inds2), axis=1)
f1 = f1 | (y1(x1), y1_obs)
f2 = f2 | (y2(x2), y2_obs)

# Predict first output.
mean1, lower1, upper1 = f1(x).marginals()

# Predict second output with Monte Carlo.
samples = [f2(B.stack(x, f1(x).sample()[:, 0], axis=1)).sample()[:, 0]
           for _ in range(100)]
mean2 = np.mean(samples, axis=0)
lower2 = np.percentile(samples, 2.5, axis=0)
upper2 = np.percentile(samples, 100 - 2.5, axis=0)

# Plot result.
plt.figure()

plt.subplot(2, 1, 1)
plt.title('Output 1')
plt.plot(x, f1_true, label='True', c='tab:blue')
plt.scatter(x_obs1, y1_obs, label='Observations', c='tab:red')
plt.plot(x, mean1, label='Prediction', c='tab:green')
plt.plot(x, lower1, ls='--', c='tab:green')
plt.plot(x, upper1, ls='--', c='tab:green')
wbml.plot.tweak()

plt.subplot(2, 1, 2)
plt.title('Output 2')
plt.plot(x, f2_true, label='True', c='tab:blue')
plt.scatter(x_obs2, y2_obs, label='Observations', c='tab:red')
plt.plot(x, mean2, label='Prediction', c='tab:green')
plt.plot(x, lower2, ls='--', c='tab:green')
plt.plot(x, upper2, ls='--', c='tab:green')
wbml.plot.tweak()

plt.savefig('readme_example7_gpar.png')
plt.show()
```

### A GP-RNN Model

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_example8_gp-rnn.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wbml.plot
from varz.tensorflow import Vars, minimise_adam
from wbml.net import rnn as rnn_constructor

from stheno.tensorflow import B, Graph, GP, Delta, EQ, Obs

# Increase regularisation because we are dealing with float32.
B.epsilon = 1e-6

# Construct points which to predict at.
x = B.linspace(tf.float32, 0, 1, 100)[:, None]
inds_obs = np.arange(0, int(0.75 * len(x)))  # Train on the first 75% only.
x_obs = B.take(x, inds_obs)

# Construct function and observations.
#   Draw random modulation functions.
a_true = GP(1e-2 * EQ().stretch(0.1))(x).sample()
b_true = GP(1e-2 * EQ().stretch(0.1))(x).sample()
#   Construct the true, underlying function.
f_true = (1 + a_true) * B.sin(2 * np.pi * 7 * x) + b_true
#   Add noise.
y_true = f_true + 0.1 * B.randn(*f_true.shape)

# Normalise and split.
f_true = (f_true - B.mean(y_true)) / B.std(y_true)
y_true = (y_true - B.mean(y_true)) / B.std(y_true)
y_obs = B.take(y_true, inds_obs)


def model(vs):
    g = Graph()

    # Construct an RNN.
    f_rnn = rnn_constructor(output_size=1,
                            widths=(10,),
                            nonlinearity=B.tanh,
                            final_dense=True)

    # Set the weights for the RNN.
    num_weights = f_rnn.num_weights(input_size=1)
    weights = Vars(tf.float32, source=vs.get(shape=(num_weights,), name='rnn'))
    f_rnn.initialise(input_size=1, vs=weights)

    # Construct GPs that modulate the RNN.
    a = GP(1e-2 * EQ().stretch(vs.pos(0.1, name='a/scale')), graph=g)
    b = GP(1e-2 * EQ().stretch(vs.pos(0.1, name='b/scale')), graph=g)
    e = GP(vs.pos(1e-2, name='e/var') * Delta(), graph=g)

    # GP-RNN model:
    f_gp_rnn = (1 + a) * (lambda x: f_rnn(x)) + b
    y_gp_rnn = f_gp_rnn + e

    return f_rnn, f_gp_rnn, y_gp_rnn, a, b


def objective_rnn(vs):
    f_rnn, _, _, _, _ = model(vs)
    return B.mean((f_rnn(x_obs) - y_obs) ** 2)


def objective_gp_rnn(vs):
    _, _, y_gp_rnn, _, _ = model(vs)
    evidence = y_gp_rnn(x_obs).logpdf(y_obs)
    return -evidence


# Pretrain the RNN.
vs = Vars(tf.float32)
minimise_adam(tf.function(objective_rnn, autograph=False),
              vs, rate=1e-2, iters=1000, trace=True)

# Jointly train the RNN and GPs.
minimise_adam(tf.function(objective_gp_rnn, autograph=False),
              vs, rate=1e-3, iters=1000, trace=True)

_, f_gp_rnn, y_gp_rnn, a, b = model(vs)

# Condition.
f_gp_rnn, a, b = (f_gp_rnn, a, b) | Obs(y_gp_rnn(x_obs), y_obs)

# Predict and plot results.
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.title('$(1 + a)\\cdot {}$RNN${} + b$')
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
mean, lower, upper = f_gp_rnn(x).marginals()
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
wbml.plot.tweak()

plt.subplot(2, 2, 3)
plt.title('$a$')
mean, lower, upper = a(x).marginals()
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
wbml.plot.tweak()

plt.subplot(2, 2, 4)
plt.title('$b$')
mean, lower, upper = b(x).marginals()
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
wbml.plot.tweak()

plt.savefig(f'readme_example8_gp-rnn.png')
plt.show()
```

### Approximate Multiplication Between GPs

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_example9_product.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import wbml.plot

from stheno import GP, EQ, model, Obs

# Define points to predict at.
x = np.linspace(0, 10, 100)

# Construct a prior.
f1 = GP(EQ(), 3)
f2 = GP(EQ(), 3)

# Compute the approximate product.
f_prod = f1 * f2

# Sample two functions.
s1, s2 = model.sample(f1(x), f2(x))

# Predict.
mean, lower, upper = (f_prod | ((f1(x), s1), (f2(x), s2)))(x).marginals()

# Plot result.
plt.plot(x, s1, label='Sample 1', c='tab:red')
plt.plot(x, s2, label='Sample 2', c='tab:blue')
plt.plot(x, s1 * s2, label='True product', c='tab:orange')
plt.plot(x, mean, label='Approximate posterior', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
wbml.plot.tweak()

plt.savefig('readme_example9_product.png')
plt.show()
```

### Sparse Regression

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_example10_sparse.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import wbml.out
import wbml.plot

from stheno import GP, EQ, Delta, SparseObs

# Define points to predict at.
x = np.linspace(0, 10, 100)
x_obs = np.linspace(0, 7, 50_000)
x_ind = np.linspace(0, 10, 20)

# Construct a prior.
f = GP(EQ().periodic(2 * np.pi))  # Latent function.
e = GP(Delta())  # Noise.
y = f + .5 * e

# Sample a true, underlying function and observations.
f_true = np.sin(x)
y_obs = np.sin(x_obs) + .5 * np.random.randn(*x_obs.shape)

# Now condition on the observations to make predictions.
obs = SparseObs(f(x_ind),  # Inducing points.
                .5 * e,  # Noise process.
                # Observations _without_ the noise process added on.
                f(x_obs), y_obs)
wbml.out.kv('elbo', obs.elbo)
mean, lower, upper = (f | obs)(x).marginals()

# Plot result.
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
plt.scatter(x_ind, 0 * x_ind, label='Inducing Points', c='black')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
wbml.plot.tweak()

plt.savefig('readme_example10_sparse.png')
plt.show()
```

### Smoothing with Nonparametric Basis Functions

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_example11_nonparametric_basis.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import wbml.plot

from stheno import GP, EQ, Delta, model, Obs

# Define points to predict at.
x = np.linspace(0, 10, 100)
x_obs = np.linspace(0, 10, 20)

# Constuct a prior:
w = lambda x: np.exp(-x ** 2 / 0.5)  # Window
b = [(GP(EQ()) * w).shift(xi) for xi in x_obs]  # Weighted basis functions
f = sum(b)  # Latent function
e = GP(Delta())  # Noise
y = f + 0.2 * e  # Observation model

# Sample a true, underlying function and observations.
f_true, y_obs = model.sample(f(x), y(x_obs))

# Condition on the observations to make predictions.
obs = Obs(y(x_obs), y_obs)
f, b = (f | obs, b | obs)

# Plot result.
for i, bi in enumerate(b):
    mean, lower, upper = bi(x).marginals()
    kw_args = {'label': 'Basis functions'} if i == 0 else {}
    plt.plot(x, mean, c='tab:orange', **kw_args)
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
mean, lower, upper = f(x).marginals()
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
wbml.plot.tweak()

plt.savefig('readme_example11_nonparametric_basis.png')
plt.show()
```

