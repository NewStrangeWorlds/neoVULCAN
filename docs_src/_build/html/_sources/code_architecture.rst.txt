Code architecture
=================

This page sketches the layout of the neoVULCAN source tree and the
responsibilities of each module. Class- and function-level documentation
generated from docstrings is in :doc:`api`.

Top-level layout
----------------

::

   neoVULCAN/
   ├── vulcan.py                # command-line driver
   ├── vulcan_api.py            # library API for embedding
   ├── vulcan_cfg.toml          # the live TOML configuration
   ├── vulcan_cfg_defaults.toml # auto-generated schema reference (do not edit)
   ├── requirements.txt
   ├── src/
   │   ├── neovulcan_config.py  # Pydantic TOML schema (VulcanConfig)
   │   ├── neovulcan_runtime.py # process-wide config singleton
   │   ├── make_chemistry_jax.py# code-generator for chemistry_jax.py
   │   ├── chemistry_jax.py     # auto-generated chemistry kernel
   │   ├── build_atm.py
   │   ├── rates.py
   │   ├── ode_solver.py / ros2.py / rodas3.py
   │   ├── integration.py
   │   ├── jacobian_jax.py
   │   ├── radiative_transfer.py    # TwoStreamRT + DisortRT
   │   ├── condensation.py
   │   ├── output.py / store.py / phy_const.py
   │   └── diagnose.py
   ├── atm/                     # T-P, Kzz and stellar-flux input
   ├── thermo/                  # network files, NASA-9 polynomials,
   │                            # photolysis cross-sections
   ├── cfg_examples/            # ready-to-run TOML configurations (.cfg)
   ├── tests/                   # regression and unit tests
   ├── plot_py/                 # plotting helpers
   └── output/                  # default destination for .vul files

Entry points
------------

``vulcan.py``
    The command-line driver. Parses ``-c`` and ``-n`` arguments, loads
    the TOML configuration through
    :class:`neovulcan_config.VulcanConfig.from_toml`, installs it as the
    process-wide singleton via :func:`neovulcan_runtime.set_cfg`, runs
    the setup pipeline (atmosphere, network, initial abundances, solver,
    radiative transfer) and then invokes the integration loop. See
    :doc:`running` for the stage-by-stage description.

``vulcan_api.py``
    A class :class:`vulcan_api.VulcanChemistry` that performs the same
    setup pipeline but exposes per-time-step methods
    (``set_atmosphere``, ``run_to_convergence``, ``get_mixing_ratios``,
    ``get_convergence_info``) suitable for coupling to a GCM. It takes
    a ``config_path`` and a ``cfg_overrides`` dict (the latter nested
    by TOML section name); the overrides are deep-merged on top of the
    loaded file and re-validated by Pydantic.

``src/make_chemistry_jax.py``
    Reads the network file, applies stoichiometric algebra symbolically
    through SymPy, and emits ``src/chemistry_jax.py`` — a self-contained
    JAX module exposing the layer-wise right-hand side
    ``chemdf(y, M, k)``, the per-layer Jacobian ``chem_jac_blocks``, the
    Gibbs free energies, and the species metadata. This is the only
    place where SymPy is used; the production solver does not depend on
    it. It is invoked automatically by ``vulcan.py`` (and by
    ``vulcan_api.VulcanChemistry.initialize`` with
    ``regenerate_chemistry=True``) unless ``-n`` is passed.


Configuration system
--------------------

The configuration is a typed Pydantic model rather than a free-form
Python module.

:mod:`neovulcan_config`
    Declares :class:`~neovulcan_config.VulcanConfig` and the ten section
    sub-models (:class:`~neovulcan_config.NetworkConfig`,
    :class:`~neovulcan_config.PathsConfig`,
    :class:`~neovulcan_config.ElementsConfig`,
    :class:`~neovulcan_config.AtmosphereConfig`,
    :class:`~neovulcan_config.PhotochemConfig`,
    :class:`~neovulcan_config.BoundaryConfig`,
    :class:`~neovulcan_config.CondensationConfig`,
    :class:`~neovulcan_config.SolverConfig`,
    :class:`~neovulcan_config.OutputConfig`,
    :class:`~neovulcan_config.PlottingConfig`). Each model uses
    ``extra='forbid'`` so typos and obsolete keys raise. Validators
    handle defaults that depend on other fields
    (``solver.dt_max`` ← ``runtime * 1e-5``, ``condensation.fix_species_time``
    ← ``stop_conden_time``, ``atmosphere.para_anaTP`` ← ``para_warm``,
    ``plotting.save_movie_rate`` ← ``live_plot_frq``) and cross-section
    rules (``P_t < P_b``, ``Rodas3 ⇒ use_moldiff``,
    ``use_ion ⇒ use_photo``). A read-only property
    :attr:`AtmosphereConfig.sl_angle` returns
    ``math.radians(sl_angle_deg)``.

:mod:`neovulcan_runtime`
    A tiny holder for the loaded :class:`~neovulcan_config.VulcanConfig`
    instance: :func:`~neovulcan_runtime.set_cfg`,
    :func:`~neovulcan_runtime.get_cfg`,
    :func:`~neovulcan_runtime.clear_cfg`, and
    :func:`~neovulcan_runtime.get_cfg_or_load` for stand-alone scripts.
    Every module under ``src/`` calls ``get_cfg()`` at import time and
    keeps a module-level handle to the section it cares about, so
    parameters are typed, tab-completable, and shared across modules
    without import-order surprises.


``src/`` modules
----------------

:mod:`store`
    Three lightweight data classes. :class:`store.Variables` holds the
    chemical state (``y``, ``ymix``), rate coefficients, photolysis
    rates, evolution arrays and element-loss diagnostics.
    :class:`store.AtmData` holds the static atmospheric structure
    (``pco``, ``Tco``, ``Kzz``, ``Dzz``, ``dz``, ``dzi``) and the
    boundary-condition data. :class:`store.Parameters` holds the
    solver counters, convergence flags and tableau metadata.

:mod:`phy_const`
    Physical constants in CGS (Boltzmann constant, Avogadro's number,
    :math:`h c`, the astronomical unit, solar radius) and the
    asymmetry factor ``ag0`` used by the two-stream solver.

:mod:`build_atm`
    Atmospheric grid construction. The class :class:`build_atm.Atm`
    builds the log-spaced pressure grid, loads (or analytically
    constructs) the T–P and :math:`\Kzz` profiles, computes the mean
    molecular weight, scale height and layer thickness, evaluates
    binary molecular-diffusion coefficients via gas-kinetic
    tabulations, and parses the boundary-condition flux files. The
    class :class:`build_atm.InitialAbun` provides initial conditions
    from FastChem (``ini_fc``) or from user input.

:mod:`rates`
    The class :class:`rates.ReadRate` parses the network file into
    elementary, three-body, special, condensation, radiative,
    photochemical and ionisation sections; evaluates all thermal rate
    coefficients in the modified Arrhenius form on the grid; computes
    reverse rates from the equilibrium constants implied by the NASA-9
    Gibbs free energies; and builds the photolysis machinery
    (wavelength binning, cross-section loading, the integral kernel
    that turns the actinic flux into J-values).

:mod:`chemistry_jax`
    Auto-generated. Exposes ``chemdf(y, M, k)`` (the chemistry RHS,
    ``vmap``-ed over layers), ``chem_jac_blocks`` (per-layer Jacobian
    via ``jax.jacfwd``), ``Gibbs(i, T)`` (equilibrium constants), and
    network metadata (``spec_list``, ``ni``, ``nr``). The module
    configures JAX for 64-bit precision and CPU execution; change
    these settings in ``make_chemistry_jax.py`` if you want different
    behaviour. The file also contains dormant infrastructure for a
    future log-space exponential-Rosenbrock integrator
    (``chemdf_logy``, ``_jac_logy_*``).

:mod:`jacobian_jax`
    Assembly of the full LHS Jacobian for the Rosenbrock W-matrix.
    ``_lhs_jac_banded_kernel(y, M, k, c0, atm_arrays)`` fuses the
    chemistry block (from ``chem_jac_blocks``), the :math:`c_0\,I`
    term, the eddy and molecular diffusion blocks, and the
    boundary-condition rows into a single banded matrix in LAPACK
    format. The kernel is JIT-compiled; per-instance caches of the
    JAX-converted atmospheric arrays reduce conversion overhead.

:mod:`ode_solver`
    Base class :class:`ode_solver.ODESolver` providing common spatial
    discretisation and step-control helpers. Computes the diffusion
    coefficients (``_eddy_coeffs``, ``_mol_diff_coeffs``), the
    transport RHS (``diffdf``, ``diffdf_settling``, ``diffdf_no_mol``,
    ``diffdf_vm``), the banded Jacobian (``lhs_jac_banded``), and the
    step-control logic (``step_ok``, ``step_reject``, ``step_size``,
    ``clip``). Holds the
    :class:`~radiative_transfer.RadiativeTransfer` instance produced by
    :func:`radiative_transfer.make_rt`.

:mod:`ros2`
    Second-order Rosenbrock W-method. The class ``Ros2`` overrides
    ``solver`` and ``solver_fix_all_bot`` with the two-stage update
    described in :doc:`numerics`, with LU reuse across stages.

:mod:`rodas3`
    Third-order, L-stable Rosenbrock–Wanner method. Implements the
    four-stage update with re-used :math:`f_2` and the embedded
    second-order error estimate. Currently supports only the standard
    transport configuration (no settling, no mixing-length model).

:mod:`integration`
    The class :class:`integration.Integration` drives the time-stepping
    loop. Responsibilities: invoking the solver per step, deciding
    when to update the radiative transfer, applying condensation
    relaxation, adjusting ``rtol`` based on element conservation,
    enforcing the diffusion-limited escape boundary condition,
    recording history, optionally firing the Newton finisher
    (``solver.use_newton_finisher``), and checking the steady-state
    criteria (``conv()``, ``stop()``).

:mod:`radiative_transfer`
    Two radiative-transfer backends and a factory that selects between
    them based on ``photochemistry.rt_scheme``:

    * :func:`~radiative_transfer.make_rt` — returns either a
      :class:`~radiative_transfer.TwoStreamRT` or a
      :class:`~radiative_transfer.DisortRT` instance.
    * :class:`~radiative_transfer.RadiativeTransfer` — a
      ``runtime_checkable`` Protocol describing the
      ``rt(var, atm) -> None`` interface that both backends honour.
    * :class:`~radiative_transfer.TwoStreamRT` — pure-NumPy
      delta-Eddington two-stream solver. Implements ``_compute_tau``,
      ``_compute_flux``, ``_compute_J`` and ``_compute_Jion``; the
      latter three are called in sequence by ``__call__``.
    * :class:`~radiative_transfer.DisortRT` — subclasses
      ``TwoStreamRT`` and overrides only ``_compute_flux``; the flux
      step delegates to the C++ DisORT++ solver imported as
      ``disortpp``. Per-bin solves run inside a single batched
      ``solve_flux_spectral`` call.

:mod:`condensation`
    The class :class:`condensation.Condensation` updates the forward
    and reverse condensation rates according to Equation
    :math:numref:`eq-cond` of :doc:`math_background`. Optional implicit
    relaxation methods for H\ :sub:`2`\ O and NH\ :sub:`3` accelerate
    convergence when the system is close to saturation.

:mod:`output`
    File I/O and console / matplotlib reporting:
    :class:`output.Output` writes the configuration and the final
    pickle, prints periodic progress and convergence summaries, and
    drives optional live plotting.

:mod:`diagnose`
    Solver-diagnostic helpers used by the regression tests and by the
    optional in-run printout. Not needed for a normal production run.


Data flow during one step
-------------------------

A successful time step calls the following pieces in order:

1. ``ode_solver.diffdf*`` evaluates the transport RHS at the current
   number densities.
2. ``chemistry_jax.chemdf`` evaluates the chemistry RHS, including the
   stored photolysis-rate coefficients ``k_J``.
3. ``jacobian_jax._lhs_jac_banded_kernel`` assembles the banded
   :math:`(I - c_0\,h\,J)` matrix.
4. ``scipy.linalg.lapack.dgbtrf`` factorises it once.
5. The selected Rosenbrock scheme calls ``dgbtrs`` for each stage and
   forms the candidate :math:`\mathbf{n}_{k+1}`.
6. ``ode_solver.step_ok`` checks the truncation error, positivity, and
   element conservation. If the step is rejected, ``step_reject``
   shrinks :math:`\Delta t` and the process restarts.
7. On an accepted step, ``ode_solver.step_size`` selects the next
   :math:`\Delta t` and ``integration.Integration`` records the new
   state.
8. Every ``photochemistry.ini_update_photo_frq`` (or
   ``final_update_photo_frq``) steps, the radiative-transfer object —
   ``TwoStreamRT`` or ``DisortRT`` depending on ``rt_scheme`` — is
   rerun and the photolysis rates updated.
9. If ``solver.use_newton_finisher`` is on and the per-step fractional
   change has dropped below ``solver.newton_switch_dy``, a short
   damped-Newton tail polishes the residual before the Rosenbrock loop
   resumes (with a ``newton_cooldown``-step cool-down).


Auxiliary directories
---------------------

``atm/``
    Pre-computed T–P profiles (e.g. ``atm_HD189_Kzz.txt``,
    ``atm_Earth_Jan_Kzz.txt``), boundary-condition flux files
    (``BC_bot_Earth.txt``, ``BC_top_Jupiter.txt``), and stellar fluxes
    (``stellar_flux/``).

``thermo/``
    Reaction network files (``NCHO_photo_network.txt``,
    ``SNCHO_full_photo_network.txt``, ``SNCHO_photo_network_2025.txt``,
    ``SO3-H2SO4_mechanism.txt``, …), Gibbs free-energy data and NASA-9
    polynomials (``gibbs_text.txt``, ``NASA9/``), photolysis
    cross-sections (``photo_cross/``).

``cfg_examples/``
    Reference TOML configurations (``Earth.cfg``, ``HD189.cfg``,
    ``Jupiter.cfg``) plus a copy of ``vulcan_cfg_defaults.toml`` for
    convenience. The ``.cfg`` extension is purely conventional;
    contents are TOML.

``tests/``
    Regression and unit tests, plus two documentation files that are
    of independent interest:

    * ``integrator_attempts_history.md`` — a curated log of integrators
      tried in neoVULCAN (log-space, naive exponential Euler, IMEX
      Rosenbrock splittings, PI step-size control). For each it
      records what was attempted, why it failed, and the chosen
      remedy. Read this before proposing a new integrator.
    * ``etd_w2_derivation.md`` — derivation of candidate exponential
      time-stepping schemes (currently dormant in the code base; the
      JAX helpers ``phi_1``, ``_jac_logy_*`` exist as scaffolding).

``plot_py/``
    Scripts to plot the contents of a ``.vul`` file (mixing ratios,
    fluxes, evolution histories). They load the configuration through
    :func:`neovulcan_runtime.get_cfg_or_load` so they can be run
    stand-alone, without going through ``vulcan.py``.
