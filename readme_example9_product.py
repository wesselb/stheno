import matplotlib.pyplot as plt
from wbml.plot import tweak

from stheno import B, Measure, GP, EQ

# Define points to predict at.
x = B.linspace(0, 10, 100)

# Construct a prior.
prior = Measure()
f1 = GP(3, EQ(), measure=prior)
f2 = GP(3, EQ(), measure=prior)

# Compute the approximate product.
f_prod = f1 * f2

# Sample two functions.
s1, s2 = prior.sample(f1(x), f2(x))

# Predict.
post = prior | ((f1(x), s1), (f2(x), s2))
mean, lower, upper = post(f_prod(x)).marginals()

# Plot result.
plt.plot(x, s1, label="Sample 1", style="train")
plt.plot(x, s2, label="Sample 2", style="train", ls="--")
plt.plot(x, s1 * s2, label="True product", style="test")
plt.plot(x, mean, label="Approximate posterior", style="pred")
plt.fill_between(x, lower, upper, style="pred")
tweak()

plt.savefig("readme_example9_product.png")
plt.show()
