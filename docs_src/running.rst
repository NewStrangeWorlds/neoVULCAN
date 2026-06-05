Running neoVULCAN
=================

There are two ways to drive the code: the standalone command-line entry
point ``vulcan.py``, and the library API in ``vulcan_api.py`` for
embedding the chemistry in a larger atmospheric model.

Command-line use
----------------

The script ``vulcan.py`` performs a complete simulation in the following
stages:

1. Parse command-line flags (``-c``, ``-n``), ``chdir`` into the
   ``vulcan.py`` directory, and prepend ``src/`` to ``sys.path``.
2. Load the TOML configuration through
   :class:`neovulcan_config.VulcanConfig.from_toml` and install it as the
   process-wide singleton via :func:`neovulcan_runtime.set_cfg`.
3. Regenerate ``src/chemistry_jax.py`` from the network file referenced
   in ``[network]`` unless ``-n`` was passed.
4. Build the atmosphere (:class:`build_atm.Atm`): pressure grid,
   temperature and :math:`\Kzz` profile, mean molecular weight, layer
   thickness, molecular diffusion coefficients, boundary-condition
   fluxes.
5. Parse the reaction network and evaluate all thermal rate coefficients
   on the grid (:class:`rates.ReadRate`).
6. Set the initial number densities (:class:`build_atm.InitialAbun`) from
   FastChem equilibrium, a constant mixing ratio, a saved ``.vul`` file,
   or a tabulated initial condition.
7. Instantiate the chosen ODE solver (``Ros2`` or ``Rodas3``).
8. If photochemistry is enabled, bin the stellar flux, load the
   photolysis cross-sections, construct the radiative-transfer object
   selected by ``photochemistry.rt_scheme`` (two-stream or DisORT++), and
   call it once.
9. Run the integration loop (:class:`integration.Integration`) until
   either the steady-state criteria are met or one of the hard limits
   (``count_max``, ``runtime``) is reached.
10. Pickle the final state to ``<output_dir>/<out_name>``.

The driver itself is a thin orchestration layer; almost all of the work
is done inside the modules in ``src/`` documented in
:doc:`code_architecture`.

Command-line options
~~~~~~~~~~~~~~~~~~~~

============================ =================================================
Flag                         Purpose
============================ =================================================
``-c PATH``,                 Path to the TOML configuration file. Default
``--config PATH``            ``vulcan_cfg.toml`` next to ``vulcan.py``.
``-n``,                      Skip the regeneration of
``--no-remake-chemistry``    ``src/chemistry_jax.py``. Use this when the
                             network file has not changed since the last run;
                             it noticeably reduces start-up time.
============================ =================================================


Library API
-----------

For coupling neoVULCAN to a three-dimensional general-circulation model
(GCM) or another driver, use :class:`vulcan_api.VulcanChemistry`. A
typical GCM time step looks like

.. code-block:: python

   import sys
   sys.path.insert(0, '/path/to/VULCAN')
   from neoVULCAN.vulcan_api import VulcanChemistry

   BASE = '/path/to/neoVULCAN'
   chem = VulcanChemistry(
       BASE,
       config_path=f'{BASE}/cfg_examples/HD189.cfg',
       cfg_overrides={
           'photochemistry': {'use_photo': True, 'rt_scheme': 'disort'},
           'solver':         {'rtol': 0.5},
       },
   )
   chem.initialize(regenerate_chemistry=True)

   for step in range(n_steps):
       T_new, P_new, Kzz_new = gcm.get_profiles(col)
       chem.set_atmosphere(T=T_new, P=P_new, Kzz=Kzz_new)
       chem.run_to_convergence()

       ymix = chem.get_mixing_ratios()         # shape (nz, ni)
       info = chem.get_convergence_info()
       gcm.update_chemistry(col, ymix, chem.species)

``cfg_overrides`` is a **nested dict** that follows the TOML structure
exactly: top-level keys are section names (``solver``, ``atmosphere``,
…), each holding a sub-dict of field overrides. It is deep-merged on top
of the loaded TOML and then re-validated by Pydantic, so the same
strictness applies to programmatic overrides as to file values.

Because ``src/chemistry_jax.py`` bakes in the species list and the
number of layers ``atmosphere.nz``, only one
:class:`~vulcan_api.VulcanChemistry` instance with a given network and
grid can exist in the same Python process at a time.


The configuration singleton
---------------------------

Internally, both entry points install the loaded
:class:`neovulcan_config.VulcanConfig` into a process-wide singleton in
:mod:`neovulcan_runtime`. Every module under ``src/`` reads its
parameters through

.. code-block:: python

   from neovulcan_runtime import get_cfg
   cfg = get_cfg()
   nz   = cfg.atmosphere.nz
   rtol = cfg.solver.rtol

so the parameters are typed (Pydantic objects), tab-completable, and
shared between modules without import-order surprises.

Stand-alone helper scripts under ``plot_py/``, ``atm/`` and ``tools/``
that are *not* launched through ``vulcan.py`` use
:func:`neovulcan_runtime.get_cfg_or_load`, which lazily loads
``vulcan_cfg.toml`` (or a path you pass) the first time it is called.


Choosing the integrator
-----------------------

Two integrators are exposed via the ``solver.ode_solver`` parameter:

``Ros2``
    Second-order Rosenbrock W-method. A-stable, two stages, two
    right-hand-side evaluations per step, one LU factorisation re-used
    across both stages. Robust default. See :doc:`numerics`.

``Rodas3``
    Third-order Rosenbrock–Wanner method. L-stable and stiffly
    accurate, with an embedded second-order error estimate. Four
    stages, three RHS evaluations, four banded back-substitutions.
    Costs roughly twice as much per step as ``Ros2`` but often takes
    fewer steps to converge. Requires ``atmosphere.use_moldiff = true``
    (the Pydantic validator rejects the combination otherwise).

Empirically, ``Ros2`` is preferred for the canonical hot-Jupiter and
terrestrial set-ups; ``Rodas3`` becomes attractive when very tight
convergence is required or when the chemistry is unusually stiff.

An optional **Newton finisher** can be enabled with
``solver.use_newton_finisher = true``; it switches from Rosenbrock to a
short damped-Newton tail once the per-step change drops below
``solver.newton_switch_dy`` to polish the residual without growing the
step count. See :doc:`config_reference` for the associated tuning knobs.


Choosing the radiative-transfer scheme
--------------------------------------

The ``photochemistry.rt_scheme`` parameter selects between two RT
backends:

``"two-stream"`` *(default)*
    Delta-Eddington two-stream solver implemented in pure NumPy
    (:class:`radiative_transfer.TwoStreamRT`). Fast, robust, and good
    enough for the vast majority of exoplanet runs. The Eddington
    coefficient is set by ``photochemistry.edd`` (default 0.5).

``"disort"``
    Per-bin discrete-ordinates solver based on the DisORT++ Python
    bindings (:class:`radiative_transfer.DisortRT`). The number of
    streams is set by ``photochemistry.disort_nstr`` (default 8);
    ``photochemistry.surface_albedo`` is honoured for the lower
    boundary. Requires the ``disortpp`` package; the import is deferred
    to ``DisortRT.__init__`` so two-stream runs work without it.

Both schemes share the optical-depth assembly and the J / J_ion
spectral integrals — only the flux step is replaced — so output arrays
have the same shape and the rest of the pipeline (convergence checks,
plotting) is unchanged. See :doc:`math_background` for the physics and
:doc:`numerics` for the implementation.


Performance notes
-----------------

* The JAX kernels (``chemistry_jax.py`` and ``jacobian_jax.py``) are
  ``jit``-compiled on the first call. The first time step of a run
  therefore pays a one-off cost of a few seconds; subsequent steps are
  fast.
* The Jacobian is stored in LAPACK banded format and factorised once
  per time step with ``dgbtrf`` / ``dgbtrs``, which is markedly faster
  than the general ``scipy.linalg.solve_banded``.
* Radiative transfer is the next biggest cost after the linear solves.
  The update frequency switches automatically from
  ``photochemistry.ini_update_photo_frq`` to
  ``photochemistry.final_update_photo_frq`` once the chemistry is close
  to steady state. The DisORT++ backend is several times more expensive
  per call than the two-stream solver but parallelises across
  wavelength bins via OpenMP inside ``disortpp``.
* ``atmosphere.update_frq`` controls how often the layer thickness
  ``dz`` is recomputed from the mean molecular weight; for nearly
  hydrostatic configurations this can be increased without affecting
  the solution.
