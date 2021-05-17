import matplotlib.pyplot as plt
from wbml.plot import tweak

from stheno import B, Measure, GP, EQ

# Define points to predict at.
x = B.linspace(0, 10, 100)
x_obs = B.linspace(0, 10, 20)

with Measure() as prior:
    w = lambda x: B.exp(-(x ** 2) / 0.5)  # Basis function
    b = [(w * GP(EQ())).shift(xi) for xi in x_obs]  # Weighted basis functions
    f = sum(b)

# Sample a true, underlying function and observations.
f_true, y_obs = prior.sample(f(x), f(x_obs, 0.2))

# Condition on the observations to make predictions.
post = prior | (f(x_obs, 0.2), y_obs)

# Plot result.
for i, bi in enumerate(b):
    mean, lower, upper = post(bi(x)).marginal_credible_bounds()
    kw_args = {"label": "Basis functions"} if i == 0 else {}
    plt.plot(x, mean, style="pred2", **kw_args)
plt.plot(x, f_true, label="True", style="test")
plt.scatter(x_obs, y_obs, label="Observations", style="train", s=20)
mean, lower, upper = post(f(x)).marginal_credible_bounds()
plt.plot(x, mean, label="Prediction", style="pred")
plt.fill_between(x, lower, upper, style="pred")
tweak()

plt.savefig("readme_example11_nonparametric_basis.png")
plt.show()
