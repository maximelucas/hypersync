hypersynchronization
====================

**Simulate, analyse, and visualize synchronization in oscillator networks with higher-order interactions.**

`hypersynchronization <https://github.com/maximelucas/hypersynchronization>`_ is a Python library that provides a modular pipeline for
Kuramoto-type models on hypergraphs: define a right-hand side function,
plug it in, and get time series, order parameters, and phase plots.
hypersynchronization uses the power of `scipy's solve_ivp <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_
to integrate ODEs and of `XGI <https://xgi.readthedocs.io>`_ for higher-order interactions.


.. image:: _static/example_about.png
   :width: 80%
   :align: center

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Simulate
      :link: getting-started
      :link-type: doc

      Integrate any RHS function on arbitrary hypergraphs using
      explicit Euler or any scipy integrator.

   .. grid-item-card:: Analyse
      :link: api
      :link-type: doc

      Compute order parameters and automatically identify synchronization
      states: twisted, splay, cluster, and more.

   .. grid-item-card:: Visualize
      :link: api
      :link-type: doc

      Plot time series, phase distributions, and summary panels
      with a single function call.


Get started by installing with

.. code-block:: bash
   pip install hypersynchronization

hypersynchronization is released under `BSD 3-Clause License <https://github.com/maximelucas/hypersynchronization/blob/docs/LICENSE>`_.

.. toctree::
   :maxdepth: 1
   :caption: Contents
   :hidden:

   getting-started
   api
   auto_examples/index
