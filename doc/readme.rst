Stheno
======

Implementation of Gaussian processes in Python

|Build| |Coverage Status| |Latest Docs|

Example: Simple Regression
--------------------------

.. figure:: https://raw.githubusercontent.com/wesselb/stheno/master/readme_prediction.png
   :alt: Prediction

   Prediction

.. code:: python

    import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
    from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI
    from stheno.tf import GP, NoisyKernel, EQ, Kronecker, Observed, Latent
    from wbml import vars32

    # Generate some observations.
    x = np.linspace(0, 10, 50, dtype=np.float32)[:, None]
    y = np.sin(x) + .2 ** .5 * np.random.randn(*x.shape)

    # And generate some data to predict for.
    x_pred = np.linspace(0, 15, 500, dtype=np.float32)[:, None]
    f_pred = np.sin(x_pred)

    # Define the parameters of the kernel.
    s2 = vars32.positive(1.)
    scale = vars32.positive(1.)
    noise = vars32.positive(.1)

    # Construct a GP and compute the evidence of the data.
    k_data = s2 * EQ().stretch(scale)
    k_noise = noise * Kronecker()
    p = GP(NoisyKernel(k_data, k_noise))
    lml = p(Observed(x)).log_pdf(y)

    # Learn.
    s = tf.Session()
    vars32.init(s)
    SOI(-lml).minimize(s)

    # Perform predictions.
    p_posterior = p.condition(Observed(x), y)
    mean, lower, upper = s.run(p_posterior.predict(Latent(x_pred)))

    # Plot.
    x, y = x.squeeze(), y.squeeze()
    x_pred, f_pred = x_pred.squeeze(), f_pred.squeeze()
    plt.plot(x_pred, f_pred, label='True', c='tab:blue')
    plt.scatter(x, y, label='Observations', c='tab:red')
    plt.plot(x_pred, mean, label='Prediction', c='tab:green')
    plt.plot(x_pred, lower, ls='--', c='tab:green')
    plt.plot(x_pred, upper, ls='--', c='tab:green')
    plt.legend()
    plt.show()

.. |Build| image:: https://travis-ci.org/wesselb/stheno.svg?branch=master
   :target: https://travis-ci.org/wesselb/stheno
.. |Coverage Status| image:: https://coveralls.io/repos/github/wesselb/stheno/badge.svg?branch=master
   :target: https://coveralls.io/github/wesselb/stheno?branch=master
.. |Latest Docs| image:: https://img.shields.io/badge/docs-latest-blue.svg
   :target: https://stheno.readthedocs.io/en/latest
