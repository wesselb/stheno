import matplotlib.pyplot as plt
from wbml.plot import tweak

from stheno import B, Measure, GP, EQ, Delta

# Define points to predict at.
x = B.linspace(0, 10, 100)
x_obs = B.linspace(0, 10, 20)

# Constuct a prior:
prior = Measure()
w = lambda x: B.exp(-(x ** 2) / 0.5)  # Window
b = [(w * GP(EQ(), measure=prior)).shift(xi) for xi in x_obs]  # Weighted basis funs
f = sum(b)  # Latent function
e = GP(Delta(), measure=prior)  # Noise
y = f + 0.2 * e  # Observation model

# Sample a true, underlying function and observations.
f_true, y_obs = prior.sample(f(x), y(x_obs))

# Condition on the observations to make predictions.
post = prior | (y(x_obs), y_obs)

# Plot result.
for i, bi in enumerate(b):
    mean, lower, upper = post(bi(x)).marginals()
    kw_args = {"label": "Basis functions"} if i == 0 else {}
    plt.plot(x, mean, style="pred2", **kw_args)
plt.plot(x, f_true, label="True", style="test")
plt.scatter(x_obs, y_obs, label="Observations", style="train", s=20)
mean, lower, upper = post(f(x)).marginals()
plt.plot(x, mean, label="Prediction", style="pred")
plt.fill_between(x, lower, upper, style="pred")
tweak()

plt.savefig("readme_example11_nonparametric_basis.png")
plt.show()
