Quickstart
==========

This page walks through running neoVULCAN on one of the bundled
example configurations and inspecting the result. For the full list of
parameters, see :doc:`config_reference`; for the library API used inside
larger atmospheric models, see :doc:`running`.

The configuration file
----------------------

Every neoVULCAN run is driven by a single **TOML** configuration file
laid out in a fixed set of sections (``[network]``, ``[paths]``,
``[elements]``, ``[atmosphere]``, ``[photochemistry]``,
``[boundary_conditions]``, ``[condensation]``, ``[solver]``,
``[output]``, ``[plotting]``). The schema is enforced by Pydantic in
``src/neovulcan_config.py`` — unknown keys, wrong types or out-of-range
values are rejected at load time with an explicit error rather than
silently ignored.

A heavily commented reference listing every parameter and its default
is generated from the schema and shipped as
``vulcan_cfg_defaults.toml`` (in both the package root and
``cfg_examples/``). Treat it as documentation; do not edit it for
production runs.


Running an example configuration
--------------------------------

The directory ``cfg_examples/`` ships three reference setups:

* ``Earth.cfg`` — Earth-like, :math:`\mathrm{N_2}`–:math:`\mathrm{O_2}`
  atmosphere with sulphur chemistry,
* ``HD189.cfg`` — hot Jupiter HD 189733b,
* ``Jupiter.cfg`` — Jupiter with deep troposphere and stratosphere.

The ``.cfg`` extension is purely conventional; the files are TOML inside
and pass straight to ``tomllib``. Run a configuration with

.. code-block:: bash

   cd neoVULCAN
   python vulcan.py -c cfg_examples/HD189.cfg

The first invocation regenerates ``src/chemistry_jax.py`` from the
network file specified in ``[network]``. On subsequent runs (when the
network has not changed) you can skip this with ``-n``:

.. code-block:: bash

   python vulcan.py -c cfg_examples/HD189.cfg -n

If you omit ``-c``, neoVULCAN looks for ``vulcan_cfg.toml`` next to
``vulcan.py`` (this is the working file checked into the package). A
typical hot-Jupiter run on a single CPU completes in a few minutes; an
Earth run with the full S–N–C–H–O network and condensation may take
longer.


Output files
------------

When the integration finishes, a pickle file
``<output_dir>/<out_name>`` is written. It contains three nested
objects:

* ``data_var`` — number densities ``y``, mixing ratios ``ymix``,
  photolysis rate coefficients ``J``, evolution arrays ``y_time``,
  ``t_time``;
* ``data_atm`` — pressure, temperature, ``Kzz``, ``Dzz``, layer thickness
  ``dz`` and interface spacing ``dzi``;
* ``data_para`` — solver counters, convergence flags, wall-clock time.

You can load the result with the helper script ``plot_py/plot_vulcan.py``
or directly in Python:

.. code-block:: python

   import pickle
   with open('output/HD189-test.vul', 'rb') as f:
       data = pickle.load(f)
   ymix    = data['variable']['ymix']     # shape (nz, n_species)
   species = data['variable']['species']


What to look at first
---------------------

Three quantities are usually worth checking before trusting a run:

1. **Convergence flag** (``data_para.end_case``). ``1`` means a numerical
   steady state was reached by the criteria in :doc:`numerics`; any other
   value indicates the run hit ``count_max`` or ``runtime`` first.

2. **Element conservation**. The ``atom_loss`` array tracks how much each
   element has drifted relative to the initial inventory; values much
   larger than :math:`10^{-2}` usually signal that ``solver.rtol`` is
   too loose or ``solver.atol`` is too tight for some species.

3. **Mixing-ratio profiles**. Plot ``ymix`` against pressure for the
   species listed in ``plotting.plot_spec``; sudden discontinuities at
   the boundaries often point to misconfigured boundary conditions
   (``boundary_conditions.use_topflux`` /
   ``boundary_conditions.use_botflux`` /
   ``boundary_conditions.use_fix_sp_bot``).
