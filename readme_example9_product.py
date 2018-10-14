import matplotlib.pyplot as plt
import numpy as np

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
plt.legend()
plt.show()
