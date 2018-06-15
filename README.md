# [Stheno](https://github.com/wesselb/stheno)

[![Build](https://travis-ci.org/wesselb/stheno.svg?branch=master)](https://travis-ci.org/wesselb/stheno)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/stheno/badge.svg?branch=master)](https://coveralls.io/github/wesselb/stheno?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://stheno.readthedocs.io/en/latest)

Implementation of Gaussian processes in Python

See also [Stheno.jl](https://github.com/willtebbutt/Stheno.jl).


## Example: Simple Regression

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
plt.plot(x, f_true.squeeze(), label='True', c='tab:blue')
plt.scatter(x_obs, y_obs.squeeze(), label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()
plt.show()
```

## Example: Decomposition of Prediction

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
    plt.plot(x, f.squeeze(), label='True', c='tab:blue')
    if x_obs is not None:
        plt.scatter(x_obs, y_obs.squeeze(), label='Observations', c='tab:red')
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

## Example: Learn a Function, Incorporating Prior Knowledge About Its Form

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
lml = y(x_obs).log_pdf(y_obs)
SOI(-lml).minimize(s)

# Print the learned parameters.
print('alpha', s.run(alpha))
print('noise', s.run(e.var))
print('u scale', s.run(u.length_scale))
print('u variance', s.run(u.var))

# Condition on the observations to make predictions.
mean, lower, upper = s.run(f.condition(y @ x_obs, y_obs).predict(x))

# Plot result.
plt.plot(x, f_true.squeeze(), label='True', c='tab:blue')
plt.scatter(x_obs, y_obs.squeeze(), label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()
plt.show()
```

## Example: Multi-Ouput Regression

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
    plt.plot(x, f.squeeze(), label='True', c='tab:blue')
    if x_obs is not None:
        plt.scatter(x_obs, y_obs.squeeze(), label='Observations', c='tab:red')
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

## Example: Approximate Integration

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
    plt.plot(x, f.squeeze(), label='True', c='tab:blue')
    if x_obs is not None:
        plt.scatter(x_obs, y_obs.squeeze(), label='Observations', c='tab:red')
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

## Example: Bayesian Linear Regression

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
plt.plot(x, f_true.squeeze(), label='True', c='tab:blue')
plt.scatter(x_obs, y_obs.squeeze(), label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()
plt.show()
```

## Example: GPAR

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
lml1 = y1(x_obs1).log_pdf(y1_obs)
SOI(-lml1, var_list=vs1.vars).minimize(s)

lml2 = y2(np.stack((x_obs2, y1_obs[inds2]), axis=1)).log_pdf(y2_obs)
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

## Example: A GP–RNN Model

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_prediction8_gp_rnn.png)


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
a = vs_gp.pos(1.0) * GP(EQ()).stretch(vs_gp.pos(0.1))
b = vs_gp.pos(1.0) * GP(EQ()).stretch(vs_gp.pos(0.1))
e = vs_gp.pos(0.2) * GP(Delta())

# RNN-only model:
y_rnn = rnn + e

# GP-RNN model:
f_gp_rnn = (1 + a) * rnn + b
y_gp_rnn = f_gp_rnn + e

# Construct evidences.
lml_rnn = y_rnn(x_obs).log_pdf(y_obs)
lml_gp_rnn = y_gp_rnn(x_obs).log_pdf(y_obs)

# Construct optimisers and initialise.
opt_rnn = tf.train.AdamOptimizer(1e-2).minimize(
    -lml_rnn, var_list=vs_rnn.vars
)
opt_gp = SOI(-lml_gp_rnn,
             options={'disp': True, 'maxiter': 10},
             var_list=vs_gp.vars)
opt_jointly = tf.train.AdamOptimizer(1e-3).minimize(
    -lml_gp_rnn, var_list=vs_rnn.vars + vs_gp.vars
)
s.run(tf.global_variables_initializer())

# Pre-train the RNN.
for i in range(500):
    _, val = s.run([opt_rnn, lml_rnn])
    if i % 100 == 0:
        print(i, val)

# Pre-train the GPs.
opt_gp.minimize(s)

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
plt.plot(x, f_true.squeeze(), label='True', c='tab:blue')
plt.scatter(x_obs, y_obs.squeeze(), label='Observations', c='tab:red')
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
