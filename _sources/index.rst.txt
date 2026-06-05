neoVULCAN
=========

**neoVULCAN** is a one-dimensional photochemical kinetics code for planetary
and exoplanetary atmospheres. It is a refactored, JAX-accelerated descendant
of the original VULCAN code [Tsai2017]_, [Tsai2021]_ (upstream repository:
`shami-EEG/VULCAN <https://github.com/shami-EEG/VULCAN>`_), designed to
integrate the stiff transport–reaction system

.. math::

   \frac{\partial n_i}{\partial t}
   = \mathcal{P}_i - \mathcal{L}_i - \frac{\partial \phi_i}{\partial z}

forward in time until a numerical steady state is reached. Compared to the
original VULCAN, neoVULCAN

* uses **JAX** for automatic differentiation of the chemistry Jacobian and
  for vectorised rate evaluations,
* exposes a **library API** (:class:`vulcan_api.VulcanChemistry`) for embedding
  in three-dimensional general-circulation models,
* supports both the second-order Rosenbrock (``Ros2``) and a third-order,
  L-stable Rosenbrock–Wanner (``Rodas3``) integrator,
* implements photochemistry, condensation with particle settling, advection,
  molecular diffusion with thermal-diffusion drift, and a flexible C–H–N–O–S
  reaction network.

This documentation collects the mathematical background, the numerical
schemes used inside neoVULCAN, a complete reference for the configuration
file, and a tour of the code architecture.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   running

.. toctree::
   :maxdepth: 2
   :caption: Theory

   math_background
   numerics

.. toctree::
   :maxdepth: 2
   :caption: Reference

   config_reference
   code_architecture
   api

.. toctree::
   :maxdepth: 1
   :caption: Appendix

   references


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
