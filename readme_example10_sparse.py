import matplotlib.pyplot as plt
import numpy as np

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
print('elbo', obs.elbo)
mean, lower, upper = (f | obs).predict(x)

# Plot result.
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
plt.scatter(x_ind, 0 * x_ind, label='Inducing Points', c='black')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()
plt.show()
