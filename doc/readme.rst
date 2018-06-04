Stheno
======

Implementation of Gaussian processes in Python

|Build| |Coverage Status| |Latest Docs|

See also `Stheno.jl <https://github.com/willtebbutt/Stheno.jl>`__.

Example: Simple Regression
--------------------------

.. figure:: https://raw.githubusercontent.com/wesselb/stheno/master/readme_prediction.png
   :alt: Prediction

   Prediction

.. code:: python

    import matplotlib.pyplot as plt
    import numpy as np

    from stheno import GP, EQ, Delta

    # Define points to predict at.
    x = np.linspace(0, 10, 100)[:, None]
    x_obs = np.linspace(0, 7, 20)[:, None]

    # Construct a prior.
    f = GP(EQ().periodic(5.))  # Latent function.
    e = GP(Delta())  # Noise.
    y = f + .5 * e

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

Example: Decomposition of Prediction
------------------------------------

.. figure:: https://raw.githubusercontent.com/wesselb/stheno/master/readme_prediction2.png
   :alt: Prediction

   Prediction

.. code:: python

    import matplotlib.pyplot as plt
    import numpy as np

    from stheno import GP, model, EQ, RQ, Linear, Delta, Exp

    # Define points to predict at.
    x = np.linspace(0, 10, 200)[:, None]
    x_obs = np.linspace(0, 7, 50)[:, None]

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
    y = f + 0.5 * e

    # Component by component, sample a true, underlying function and observations.
    f_true_smooth = f_smooth(x).sample()
    model.condition(f_smooth @ x, f_true_smooth)

    f_true_wiggly = f_wiggly(x).sample()
    model.condition(f_wiggly @ x, f_true_wiggly)

    f_true_periodic = f_periodic(x).sample()
    model.condition(f_periodic @ x, f_true_periodic)

    f_true_linear = f_linear(x).sample()
    model.condition(f_linear @ x, f_true_linear)

    f_true = f(x).sample()
    model.condition(f @ x, f_true)

    y_obs = y(x_obs).sample()
    model.revert_prior()

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
        plt.plot(x.squeeze(), f.squeeze(), label='True', c='tab:blue')
        if x_obs is not None:
            plt.scatter(x_obs.squeeze(), y_obs.squeeze(),
                        label='Observations', c='tab:red')
        mean, lower, upper = pred
        plt.plot(x.squeeze(), mean, label='Prediction', c='tab:green')
        plt.plot(x.squeeze(), lower, ls='--', c='tab:green')
        plt.plot(x.squeeze(), upper, ls='--', c='tab:green')
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

Example: Learn a Function, Incorporating Prior Knowledge About Its Form
-----------------------------------------------------------------------

.. figure:: https://raw.githubusercontent.com/wesselb/stheno/master/readme_prediction3.png
   :alt: Prediction

   Prediction

.. code:: python

    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI
    from wbml import vars64 as vs

    from stheno.tf import GP, EQ, Delta, model

    s = tf.Session()

    # Define points to predict at.
    x = np.linspace(0, 5, 100)[:, None]
    x_obs = np.linspace(0, 3, 20)[:, None]

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
    x, f_true, x_obs, y_obs = map(np.squeeze, (x, f_true, x_obs, y_obs))
    plt.plot(x, f_true, label='True', c='tab:blue')
    plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
    plt.plot(x, mean, label='Prediction', c='tab:green')
    plt.plot(x, lower, ls='--', c='tab:green')
    plt.plot(x, upper, ls='--', c='tab:green')
    plt.legend()
    plt.show()

.. |Build| image:: https://travis-ci.org/wesselb/stheno.svg?branch=master
   :target: https://travis-ci.org/wesselb/stheno
.. |Coverage Status| image:: https://coveralls.io/repos/github/wesselb/stheno/badge.svg?branch=master
   :target: https://coveralls.io/github/wesselb/stheno?branch=master
.. |Latest Docs| image:: https://img.shields.io/badge/docs-latest-blue.svg
   :target: https://stheno.readthedocs.io/en/latest
