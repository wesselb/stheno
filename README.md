# Stheno
Implementation of Gaussian processes in Python


[![Build](https://travis-ci.org/wesselb/stheno.svg?branch=master)](https://travis-ci.org/wesselb/stheno)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/stheno/badge.svg?branch=master)](https://coveralls.io/github/wesselb/stheno?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://stheno.readthedocs.io/en/latest)


## Example: Simple Regression

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/readme_prediction.png)

```python
import matplotlib.pyplot as plt
import numpy as np

from stheno import GP, EQ, Kronecker

# Define points to predict at.
x = np.linspace(0, 10, 100)[:, None]
x_obs = np.linspace(0, 7, 10)[:, None]

# Construct a prior.
f = GP(EQ())  # Latent function.
e = GP(0.1 * Kronecker())  # Noise.
y = f + e

# Sample a true, underlying function.
f_true = f(x).sample()

# Condition the model on the true function and sample observations.
y_obs = y.condition(f @ x, f_true)(x_obs).sample()
y.revert_prior()

# Now condition on the observations to make predictions.
mean, lower, upper = f.condition(y @ x_obs, y_obs).predict(x)

# Plot result.
x, f_true, x_obs, y_obs = map(np.squeeze, (x, f_true, x_obs, y_obs))
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()
plt.show()

```

