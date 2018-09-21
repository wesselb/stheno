# [Stheno](https://github.com/wesselb/stheno)

[![Build](https://travis-ci.org/wesselb/stheno.svg?branch=master)](https://travis-ci.org/wesselb/stheno)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/stheno/badge.svg?branch=master)](https://coveralls.io/github/wesselb/stheno?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://stheno.readthedocs.io/en/latest)

Stheno is an implementation of Gaussian process modelling in Python. See 
also [Stheno.jl](https://github.com/willtebbutt/Stheno.jl).

* [Nonlinear Regression in 20 Seconds](#nonlinear-regression-in-20-seconds)
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
    - [Inference and Sampling](#inference-and-sampling)
    - [NumPy, TensorFlow, or PyTorch?](#numpy-tensorflow-or-pytorch)
    - [Undiscussed Features](#undiscussed-features)
* [Examples](#examples)
    - [Simple Regression](#simple-regression)
    - [Decomposition of Prediction](#decomposition-of-prediction)
    - [Learn a Function, Incorporating Prior Knowledge About Its Form](#learn-a-function-incorporating-prior-knowledge-about-its-form)
    - [Multi-Output Regression](#multi-ouput-regression)
    - [Approximate Integration](#approximate-integration)
    - [Bayesian Linear Regression](#bayesian-linear-regression)
    - [GPAR](#gpar)
    - [A GP–RNN Model](#a-gprnn-model)
    - [Approximate Multiplication Between GPs](#approximate-multiplication-between-gps)

## Nonlinear Regression in 20 Seconds

```python
>>> import numpy as np

>>> from stheno import GP, EQ

>>> x = np.linspace(0, 2, 10)         # Points to predict at

>>> y = x ** 2                        # Observations

>>> GP(EQ()).condition(x, y)(3).mean  # Go GP!
array([[8.48258669]])
```

Moar?! Then read on!

## Manual

Note: [here](https://stheno.readthedocs.io/en/latest) is a nicely rendered 
and more readable version of the docs.

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
matrix_ for all pairs of points between `x` and `y`. `k(x)` is shorthand for `k
(x, x)`. Furthermore, `k.elwise(x, y)` constructs the _kernel vector_ pairing
the points in `x` and `y` element-wise, which will be a _rank 2 column vector_.

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
    >>> EQ().stretch(1)
    EQ() > 1

    >>> EQ().stretch(1, 2)
    EQ() > (1, 2)
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

* Numerically, but efficiently, take derivatives of _kernels and means_. This 
currently only works in TensorFlow and derivatives cannot be nested.

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
    
* Select particular input dimensions.

    Example:
    
    ```python
    >>> GP(EQ()).select(1, 3)
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

### Inference and Sampling

To condition on observations, use `Graph.condition` or `GP.condition`.
Syntax is much like the math:
compare `f1.condition(f2 @ x, y)` with $f_1 \,|\, f_2(x) = y$.

Definition, where `f*` are `GP`s:

```python
model.condition(f @ x, y)

model.condition((f1 @ x1, y1), (f2 @ x2, y2), ...)

f1_updated = f1.condition(x, y)

f1_updated = f1.condition((f1 @ x1, y1), (f2 @ x2, y2), ...)
```

_Important:_ both `Graph.condition` and `GP.condition` are _mutative_:
once either is called, all further operations will be conditional on the given observations.
If you want to undo the conditioning operation and revert to the state _just before the first conditioning operation_, use `Graph.revert_prior` or `GP.revert_prior`.

Example:

```python
>>> f.condition(x, y)

>>> # Anything here will be conditional on `f(x) = y`.

>>> f.revert_prior()
```

Alternatively, or for more fine-grained constrol, use _checkpoints_.

Example:

```python
>>> prior = model.checkpoint()

>>> f1.condition(x, y)

>>> # Anything here will be conditional on `f1(x) = y`.

>>> conditional_on_f1 = model.checkpoint()

>>> f2.condition(x, y)

>>> # Anything here will be conditional on `f1(x) = y` and `f2(x) = y`.

>>> model.revert(conditional_on_f1)

>>> # Anything here will again be conditional on `f1(x) = y`.

>>> model.revert(prior)
```

After conditioning, simply call a GP to construct its finite-dimensional 
distribution:

```python
>>> f(x)
<stheno.random.Normal at 0x10effa080>

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
```

Alternatively, use `f.predict(x)` to efficiently compute the means and 
the marginal lower and upper 95% central credible region bounds:

```python
>>> f.predict(x)
(array([0., 0., 0.]), array([-2., -2., -2.]), array([2., 2., 2.]))
```

Finally, `Graph.sample` can be used to get samples from multiple processes 
jointly:

```python
>>> model.sample(f @ x, (2 * f) @ x)
[array([[-0.35226314],
        [-0.15521219],
        [ 0.0752406 ]]),
 array([[-0.70452827],
        [-0.31042226],
        [ 0.15048168]])]
```

### NumPy, TensorFlow, or PyTorch?

Your choice!

```python
from stheno import GP, EQ
```

```python
from stheno.tf import GP, EQ
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

    >>> prediction = p.condition(Observed(x), y).predict(Latent(x))
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
    >>> GP(EQ()) * GP(EQ())
    GP((EQ() + (ZeroMean x ZeroMean)) * (EQ() + (ZeroMean x ZeroMean)) + (ZeroMean x ZeroMean) * (ZeroMean x ZeroMean) + -2 * (ZeroMean x ZeroMean), <lambda>)
    ```

## Examples

### Simple Regression

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_prediction1_simple_regression.png)

```python
import matplotlib.pyplot as plt
import numpy as np

from stheno import GP, EQ, Delta, model

# Define points to predict at.
x = np.linspace(0, 10, 100)
x_obs = np.linspace(0, 7, 20)

# Construct a prior.
f = GP(EQ().periodic(5.))  # Latent function.
e = GP(Delta())  # Noise.
y = f + .5 * e

# Sample a true, underlying function and observations.
f_true, y_obs = model.sample(f @ x, y @ x_obs)

# Now condition on the observations to make predictions.
mean, lower, upper = f.condition(y @ x_obs, y_obs).predict(x)

# Plot result.
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()
plt.show()
```

### Decomposition of Prediction

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_prediction2_decomposition.png)

```python
import matplotlib.pyplot as plt
import numpy as np

from stheno import GP, model, EQ, RQ, Linear, Delta, Exp

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
    model.sample(f_smooth @ x,
                 f_wiggly @ x,
                 f_periodic @ x,
                 f_linear @ x,
                 f @ x,
                 y @ x_obs)

# Now condition on the observations and make predictions for the latent
# function and its various components.
model.condition(y @ x_obs, y_obs)

pred_smooth = f_smooth.predict(x)
pred_wiggly = f_wiggly.predict(x)
pred_periodic = f_periodic.predict(x)
pred_linear = f_linear.predict(x)
pred_f = f.predict(x)


# Plot results.
def plot_prediction(x, f, pred, x_obs=None, y_obs=None):
    plt.plot(x, f, label='True', c='tab:blue')
    if x_obs is not None:
        plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
    mean, lower, upper = pred
    plt.plot(x, mean, label='Prediction', c='tab:green')
    plt.plot(x, lower, ls='--', c='tab:green')
    plt.plot(x, upper, ls='--', c='tab:green')
    plt.legend()


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

plt.show()
```

### Learn a Function, Incorporating Prior Knowledge About Its Form

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_prediction3_parametric.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI
from wbml import vars64 as vs

from stheno.tf import GP, EQ, Delta, model

s = tf.Session()

# Define points to predict at.
x = np.linspace(0, 5, 100)
x_obs = np.linspace(0, 3, 20)

# Construct the model.
u = GP(vs.pos(.5) * EQ().stretch(vs.pos(1.)))
e = GP(vs.pos(.5) * Delta())
alpha = vs.pos(1.2)
vs.init(s)

f = u + (lambda x: x ** alpha)
y = f + e

# Sample a true, underlying function and observations.
f_true = x ** 1.8
y_obs = s.run(y.condition(f @ x, f_true)(x_obs).sample())
model.revert_prior()

# Learn.
lml = y(x_obs).logpdf(y_obs)
SOI(-lml).minimize(s)

# Print the learned parameters.
print('alpha', s.run(alpha))
print('noise', s.run(e.var))
print('u scale', s.run(u.length_scale))
print('u variance', s.run(u.var))

# Condition on the observations to make predictions.
mean, lower, upper = s.run(f.condition(y @ x_obs, y_obs).predict(x))

# Plot result.
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()
plt.show()
```

### Multi-Ouput Regression

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_prediction4_multi-output.png)

```python
import matplotlib.pyplot as plt
import numpy as np
from plum import Dispatcher, Referentiable, Self

from stheno import GP, EQ, Delta, model, Kernel


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
        ps = [0 for i in range(m)]
        for i in range(m):
            for j in range(n):
                ps[i] += A[i, j] * self.ps[j]
        return VGP(*ps)

    def sample(self, x):
        return model.sample(*(p @ x for p in self.ps))

    def condition(self, x, ys):
        model.condition(*((p @ x, y) for p, y in zip(self.ps, ys)))
        return self

    def predict(self, x):
        return [p.predict(x) for p in self.ps]


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

# Sample observations and a true, underlying function.
ys_obs = ys.sample(x_obs)
ys.condition(x_obs, ys_obs)
fs_true = fs.sample(x)
model.revert_prior()

# Condition the model on the observations to make predictions.
ys.condition(x_obs, ys_obs)
preds = fs.predict(x)


# Plot results.
def plot_prediction(x, f, pred, x_obs=None, y_obs=None):
    plt.plot(x, f, label='True', c='tab:blue')
    if x_obs is not None:
        plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
    mean, lower, upper = pred
    plt.plot(x, mean, label='Prediction', c='tab:green')
    plt.plot(x, lower, ls='--', c='tab:green')
    plt.plot(x, upper, ls='--', c='tab:green')
    plt.legend()


plt.figure(figsize=(10, 6))

for i in range(p):
    plt.subplot(int(p ** .5), int(p ** .5), i + 1)
    plt.title('Output {}'.format(i + 1))
    plot_prediction(x, fs_true[i], preds[i], x_obs, ys_obs[i])

plt.show()
```

### Approximate Integration

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_prediction5_integration.png)

```python
import matplotlib.pyplot as plt
import numpy as np

from stheno import GP, EQ, Delta, model

# Define points to predict at.
x = np.linspace(0, 10, 200)
x_obs = np.linspace(0, 10, 10)

# Construct the model.
f = 0.7 * GP(EQ()).stretch(1.5)
e = 0.2 * GP(Delta())

# Construct derivatives via finite differences.
df = f.diff_approx(1)
ddf = f.diff_approx(2)
dddf = f.diff_approx(3) + e

# Fix the integration constants.
model.condition((f @ 0, 1), (df @ 0, 0), (ddf @ 0, -1))

# Sample observations.
y_obs = np.sin(x_obs) + 0.2 * np.random.randn(*x_obs.shape)

# Condition on the observations to make predictions.
model.condition(dddf @ x_obs, y_obs)

# And make predictions.
pred_iiif = f.predict(x)
pred_iif = df.predict(x)
pred_if = ddf.predict(x)
pred_f = dddf.predict(x)


# Plot result.
def plot_prediction(x, f, pred, x_obs=None, y_obs=None):
    plt.plot(x, f, label='True', c='tab:blue')
    if x_obs is not None:
        plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
    mean, lower, upper = pred
    plt.plot(x, mean, label='Prediction', c='tab:green')
    plt.plot(x, lower, ls='--', c='tab:green')
    plt.plot(x, upper, ls='--', c='tab:green')
    plt.legend()


plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.title('Function')
plot_prediction(x, np.sin(x), pred_f, x_obs=x_obs, y_obs=y_obs)
plt.legend()

plt.subplot(2, 2, 2)
plt.title('Integral of Function')
plot_prediction(x, -np.cos(x), pred_if)
plt.legend()

plt.subplot(2, 2, 3)
plt.title('Second Integral of Function')
plot_prediction(x, -np.sin(x), pred_iif)
plt.legend()

plt.subplot(2, 2, 4)
plt.title('Third Integral of Function')
plot_prediction(x, np.cos(x), pred_iiif)
plt.legend()

plt.show()
```

### Bayesian Linear Regression

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_prediction6_blr.png)

```python
import matplotlib.pyplot as plt
import numpy as np

from stheno import GP, Delta, model

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
    model.sample(slope @ 0, intercept @ 0, f @ x, y @ x_obs)

# Condition on the observations to make predictions.
mean, lower, upper = f.condition(y @ x_obs, y_obs).predict(x)
mean_slope, mean_intercept = slope(0).mean, intercept(0).mean

print('true slope', true_slope)
print('predicted slope', mean_slope)
print('true intercept', true_intercept)
print('predicted intercept', mean_intercept)

# Plot result.
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()
plt.show()
```

### GPAR

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_prediction7_gpar.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI
from wbml import Vars

from stheno.tf import GP, Delta, EQ, Graph, B

s = tf.Session()

# Define points to predict at.
x = np.linspace(0, 10, 200)
x_obs1 = np.linspace(0, 10, 30)
inds2 = np.random.permutation(len(x_obs1))[:10]
x_obs2 = x_obs1[inds2]

# Construct variable storages.
vs1 = Vars(np.float64)
vs2 = Vars(np.float64)

# Construct a model for each output.
m1 = Graph()
m2 = Graph()
f1 = vs1.pos(1.) * GP(EQ(), graph=m1).stretch(vs1.pos(1.))
f2 = vs2.pos(1.) * GP(EQ(), graph=m2).stretch(vs2.pos([1., .5]))
sig1 = vs1.pos(0.1)
sig2 = vs2.pos(0.1)

# Initialise variables.
vs1.init(s)
vs2.init(s)

# Noise models:
e1 = sig1 * GP(Delta(), graph=m1)
e2 = sig2 * GP(Delta(), graph=m2)

# Observation models:
y1 = f1 + e1
y2 = f2 + e2

# Construction functions to predict and observations.
f1_true = np.sin(x)
f2_true = np.sin(x) ** 2

y1_obs = np.sin(x_obs1) + 0.1 * np.random.randn(*x_obs1.shape)
y2_obs = np.sin(x_obs2) ** 2 + 0.1 * np.random.randn(*x_obs2.shape)

# Learn.
lml1 = y1(x_obs1).logpdf(y1_obs)
SOI(-lml1, var_list=vs1.vars).minimize(s)

lml2 = y2(np.stack((x_obs2, y1_obs[inds2]), axis=1)).logpdf(y2_obs)
SOI(-lml2, var_list=vs2.vars).minimize(s)

# Predict first output.
mean1, lower1, upper1 = s.run(f1.condition(y1 @ x_obs1, y1_obs).predict(x))

# Predict second output with Monte Carlo.
m2.condition(y2 @ np.stack((x_obs2, y1_obs[inds2]), axis=1), y2_obs)
sample = f2(B.concat([x[:, None], f1(x).sample()], axis=1)).sample()
samples = [s.run(sample).squeeze() for _ in range(100)]
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
plt.legend()

plt.subplot(2, 1, 2)
plt.title('Output 2')
plt.plot(x, f2_true, label='True', c='tab:blue')
plt.scatter(x_obs2, y2_obs, label='Observations', c='tab:red')
plt.plot(x, mean2, label='Prediction', c='tab:green')
plt.plot(x, lower2, ls='--', c='tab:green')
plt.plot(x, upper2, ls='--', c='tab:green')
plt.legend()

plt.show()
```

### A GP–RNN Model

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_prediction8_gp-rnn.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI
from wbml import Vars, rnn as rnn_constructor

from stheno.tf import GP, Delta, model, EQ, RQ

# Construct variable storages.
vs_gp = Vars(np.float32)
vs_rnn = Vars(np.float32)

# Construct a 1-layer RNN with GRUs.
f_rnn = rnn_constructor(1, 1, (10,))
f_rnn.initialise(vs_rnn)


# Wrap the RNN to be compatible with Stheno.
def rnn(x):
    return f_rnn(x[:, :, None])[:, :, 0]


# Construct session.
s = tf.Session()

# Construct points which to predict at.
x = np.linspace(0, 1, 100, dtype=np.float32)
inds_obs = np.arange(0, int(0.75 * len(x)))  # Train on the first 75% only.
x_obs = x[inds_obs]

# Construct function and observations.
#   Draw a random fluctuation.
k_u = .2 * RQ(1e-1).stretch(0.05)
u = s.run(GP(k_u)(np.array(x, dtype=np.float64)).sample()).squeeze()
#   Construct the true, underlying function.
f_true = np.sin(2 * np.pi * 7 * x) + np.array(u, dtype=np.float32)
#   Add noise.
y_true = f_true + 0.2 * np.array(np.random.randn(*x.shape), dtype=np.float32)

# Normalise and split.
f_true = (f_true - np.mean(y_true)) / np.std(y_true)
y_true = (y_true - np.mean(y_true)) / np.std(y_true)
y_obs = y_true[inds_obs]

# Construct the model.
a = 0.1 * GP(EQ()).stretch(vs_gp.pos(0.1))
b = 0.1 * GP(EQ()).stretch(vs_gp.pos(0.1))
e = vs_gp.pos(0.1) * GP(Delta())

# RNN-only model:
y_rnn = rnn + e

# GP-RNN model:
f_gp_rnn = (1 + a) * rnn + b
y_gp_rnn = f_gp_rnn + e

# Construct evidences.
lml_rnn = y_rnn(x_obs).logpdf(y_obs)
lml_gp_rnn = y_gp_rnn(x_obs).logpdf(y_obs)

# Construct optimisers and initialise.
opt_rnn = tf.train.AdamOptimizer(1e-2).minimize(
    -lml_rnn, var_list=vs_rnn.vars
)
opt_jointly = tf.train.AdamOptimizer(1e-3).minimize(
    -lml_gp_rnn, var_list=vs_rnn.vars + vs_gp.vars
)
s.run(tf.global_variables_initializer())

# Nudge the RNN into the right direction.
for i in range(2000):
    _, val = s.run([opt_rnn, lml_rnn])
    if i % 100 == 0:
        print(i, val)

# Jointly train the RNN and GPs.
for i in range(5000):
    _, val = s.run([opt_jointly, lml_gp_rnn])
    if i % 100 == 0:
        print(i, val)

# Condition.
model.condition(y_gp_rnn @ x_obs, y_obs)

# Predict and plot results.
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.title('$(1 + a) \\cdot $ RNN ${}+b$')
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
mean, lower, upper = s.run(f_gp_rnn.predict(x))
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()

plt.subplot(2, 2, 3)
plt.title('$a$')
mean, lower, upper = s.run(a.predict(x))
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()

plt.subplot(2, 2, 4)
plt.title('$b$')
mean, lower, upper = s.run(b.predict(x))
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()

plt.show()
```

### Approximate Multiplication Between GPs

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_prediction9_product.png)

```python
import matplotlib.pyplot as plt
import numpy as np

from stheno import GP, EQ, model

# Define points to predict at.
x = np.linspace(0, 10, 100)

# Construct a prior.
f1 = GP(EQ(), 3)
f2 = GP(EQ(), 3)

# Compute the approximate product.
f_prod = f1 * f2

# Sample two functions.
s1, s2 = model.sample(f1 @ x, f2 @ x)

# Predict.
mean, lower, upper = f_prod.condition((f1 @ x, s1), (f2 @ x, s2)).predict(x)

# Plot result.
plt.plot(x, s1, label='Sample 1', c='tab:red')
plt.plot(x, s2, label='Sample 2', c='tab:blue')
plt.plot(x, s1 * s2, label='True product', c='tab:orange')
plt.plot(x, mean, label='Approximate posterior', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()
plt.show()
```
