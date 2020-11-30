import matplotlib.pyplot as plt
from wbml.plot import tweak

from stheno import B, Measure, GP, EQ, Delta

# Define points to predict at.
x = B.linspace(0, 10, 100)
x_obs = B.linspace(0, 7, 20)

# Construct a prior.
prior = Measure()
f = GP(EQ().periodic(5.0), measure=prior)  # Latent function
e = GP(Delta(), measure=prior)  # Noise
y = f + 0.5 * e

# Sample a true, underlying function and observations.
f_true, y_obs = prior.sample(f(x), y(x_obs))

# Now condition on the observations to make predictions.
post = prior | (y(x_obs), y_obs)
mean, lower, upper = post(f)(x).marginals()

# Plot result.
plt.plot(x, f_true, label="True", style="test")
plt.scatter(x_obs, y_obs, label="Observations", style="train", s=20)
plt.plot(x, mean, label="Prediction", style="pred")
plt.fill_between(x, lower, upper, style="pred")
tweak()
plt.savefig("readme_example1_simple_regression.png")
plt.show()
