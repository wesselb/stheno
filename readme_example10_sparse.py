import matplotlib.pyplot as plt
import wbml.out as out
from wbml.plot import tweak

from stheno import B, GP, EQ, PseudoObs

# Define points to predict at.
x = B.linspace(0, 10, 100)
x_obs = B.linspace(0, 7, 50_000)
x_ind = B.linspace(0, 10, 20)

# Construct a prior.
f = GP(EQ().periodic(2 * B.pi))

# Sample a true, underlying function and observations.
f_true = B.sin(x)
y_obs = B.sin(x_obs) + B.sqrt(0.5) * B.randn(*x_obs.shape)

# Compute a pseudo-point approximation of the posterior.
obs = PseudoObs(f(x_ind), (f(x_obs, 0.5), y_obs))

# Compute the ELBO.
out.kv("ELBO", obs.elbo(f.measure))

# Compute the approximate posterior.
f_post = f | obs

# Make predictions with the approximate posterior.
mean, lower, upper = f_post(x).marginal_credible_bounds()

# Plot result.
plt.plot(x, f_true, label="True", style="test")
plt.scatter(
    x_obs,
    y_obs,
    label="Observations",
    style="train",
    c="tab:green",
    alpha=0.35,
)
plt.scatter(
    x_ind,
    obs.mu(f.measure)[:, 0],
    label="Inducing Points",
    style="train",
    s=20,
)
plt.plot(x, mean, label="Prediction", style="pred")
plt.fill_between(x, lower, upper, style="pred")
tweak()

plt.savefig("readme_example10_sparse.png")
plt.show()
