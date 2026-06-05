API reference
=============

This section is generated from the Python docstrings of the neoVULCAN
modules. For a narrative overview of the codebase, see
:doc:`code_architecture`.

.. note::

   Many modules depend on the auto-generated ``chemistry_jax.py`` which
   in turn imports JAX. If JAX is not installed in the documentation
   build environment, the affected modules are mocked through
   ``autodoc_mock_imports`` in ``conf.py``; the documented signatures
   are still correct.


Library entry point
-------------------

.. automodule:: vulcan_api
   :members:
   :undoc-members:
   :show-inheritance:


Configuration schema and runtime
--------------------------------

.. automodule:: neovulcan_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: neovulcan_runtime
   :members:
   :undoc-members:


Data containers
---------------

.. automodule:: store
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phy_const
   :members:
   :undoc-members:


Atmosphere and initial conditions
---------------------------------

.. automodule:: build_atm
   :members:
   :undoc-members:
   :show-inheritance:


Reaction rates and photochemistry setup
---------------------------------------

.. automodule:: rates
   :members:
   :undoc-members:
   :show-inheritance:


Chemistry kernel and Jacobian
-----------------------------

.. automodule:: chemistry_jax
   :members:
   :undoc-members:

.. automodule:: jacobian_jax
   :members:
   :undoc-members:


ODE solver
----------

.. automodule:: ode_solver
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ros2
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rodas3
   :members:
   :undoc-members:
   :show-inheritance:


Integration driver
------------------

.. automodule:: integration
   :members:
   :undoc-members:
   :show-inheritance:


Radiative transfer
------------------

.. automodule:: radiative_transfer
   :members:
   :undoc-members:
   :show-inheritance:


Condensation
------------

.. automodule:: condensation
   :members:
   :undoc-members:
   :show-inheritance:


Output and I/O
--------------

.. automodule:: output
   :members:
   :undoc-members:
   :show-inheritance:
