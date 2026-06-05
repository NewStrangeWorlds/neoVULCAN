Configuration reference
=======================

A neoVULCAN simulation is configured by a single **TOML** file passed to
``vulcan.py`` with the ``-c`` flag (default: ``vulcan_cfg.toml`` next to
``vulcan.py``). The file is parsed by ``tomllib`` (Python 3.11+) or
``tomli`` (Python 3.10) and validated against the Pydantic schema
defined in ``src/neovulcan_config.py``. Validation is **strict**: any
unknown key, wrong type, or out-of-range value is rejected at load time
with a precise error message rather than silently ignored.

The file is organised into ten sections. Each section maps to one of
the Pydantic model classes
(:class:`~neovulcan_config.NetworkConfig`,
:class:`~neovulcan_config.PathsConfig`,
:class:`~neovulcan_config.ElementsConfig`,
:class:`~neovulcan_config.AtmosphereConfig`,
:class:`~neovulcan_config.PhotochemConfig`,
:class:`~neovulcan_config.BoundaryConfig`,
:class:`~neovulcan_config.CondensationConfig`,
:class:`~neovulcan_config.SolverConfig`,
:class:`~neovulcan_config.OutputConfig`,
:class:`~neovulcan_config.PlottingConfig`).

A machine-generated reference with every parameter and its default is
shipped at ``vulcan_cfg_defaults.toml`` in the package root and in
``cfg_examples/``. Treat it as documentation; do not edit it for
production runs. To regenerate it after a schema change run:

.. code-block:: bash

   python _gen_defaults_toml.py

CGS units (cm, g, s, K, dyn cm\ :sup:`-2`, erg) are used throughout
unless explicitly stated.


``[network]``
-------------

Chemical-reaction network and thermochemistry inputs.
Schema: :class:`neovulcan_config.NetworkConfig`.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Key
     - Type
     - Description
   * - ``atom_list``
     - list[str]
     - **Required.** Elements tracked for conservation (e.g.
       ``["H", "O", "C", "N", "S", "He"]``).
   * - ``network``
     - str
     - **Required.** Path to the network file (e.g.
       ``"thermo/SNCHO_photo_network_2025.txt"``).
   * - ``gibbs_text``
     - str
     - Index file listing the NASA-9 polynomial files used to build
       Gibbs free energies. Default ``"thermo/gibbs_text.txt"``.
   * - ``com_file``
     - str
     - Composition file with basic per-species properties (mass,
       charge, atomic content). Default ``"thermo/all_compose.txt"``.
   * - ``cross_folder``
     - str
     - Directory of photolysis cross-sections.
       Default ``"thermo/photo_cross/"``.
   * - ``use_lowT_limit_rates``
     - bool
     - Clip thermal rates to their low-temperature limits below the
       Arrhenius validity range. Default ``false``.
   * - ``remove_list``
     - list[int]
     - Reaction indices to disable. Forward/reverse pairs should be
       removed together (e.g. ``[315, 316]``). Default ``[]``.


``[paths]``
-----------

File-system paths for atmospheric input, stellar flux, boundary
conditions, and output.
Schema: :class:`neovulcan_config.PathsConfig`.

When ``VulcanConfig.from_toml`` is called with ``absolutify_paths=True``
(used by the library API), relative paths are resolved against the
``base_dir`` argument (TOML's directory by default); otherwise the
values are passed through unchanged.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Key
     - Type
     - Description
   * - ``fastchem_dir``
     - str
     - FastChem element & rate-coefficient directory. Default
       ``"fastchem_input"``.
   * - ``atm_file``
     - str
     - **Required.** Temperature–pressure profile (and optional
       :math:`\Kzz`).
   * - ``sflux_file``
     - str
     - **Required.** Stellar flux density at the stellar surface
       (:math:`\mathrm{erg\,cm^{-2}\,s^{-1}\,nm^{-1}}`).
   * - ``top_BC_flux_file``
     - str
     - Top-of-atmosphere boundary-condition flux file. Default
       ``"atm/"``.
   * - ``bot_BC_flux_file``
     - str
     - Bottom-of-atmosphere boundary-condition flux file. Default
       ``"atm/"``.
   * - ``vul_ini``
     - str
     - Previous ``.vul`` file to initialise abundances from (only read
       when ``elements.ini_mix = "vulcan_ini"``). Default ``"output/"``.
   * - ``output_dir``
     - str
     - Output directory for the pickle file. Default ``"output/"``.
   * - ``plot_dir``
     - str
     - Output directory for plots. Default ``"plot/"``.
   * - ``movie_dir``
     - str
     - Output directory for live-plot frames. Default
       ``"plot/movie/"``.
   * - ``out_name``
     - str
     - **Required.** Output file name (extension ``.vul``).


``[elements]``
--------------

Elemental abundances and initial mixing-ratio mode.
Schema: :class:`neovulcan_config.ElementsConfig`.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Key
     - Type
     - Description
   * - ``use_solar``
     - bool
     - Use the solar abundances of Lodders (2009). If ``false`` the
       ``metal_abundances`` sub-table is used instead. Default ``false``.
   * - ``metal_abundances``
     - dict[str, float]
     - Element-to-hydrogen number ratios (e.g.
       ``{O = 5.37e-4, C = 2.95e-4, He = 0.0838}``). Empty by default;
       set in the sub-table ``[elements.metal_abundances]``.
   * - ``ini_mix``
     - str
     - Initial-condition mode: one of ``"EQ"`` (FastChem equilibrium),
       ``"const_mix"``, ``"vulcan_ini"``, ``"table"``. Default
       ``"const_mix"``.
   * - ``fastchem_met_scale``
     - float
     - Metallicity scaling factor passed to FastChem for the trace
       elements not in ``atom_list`` (e.g. Si, Mg). ``1.0`` is solar.
   * - ``const_mix``
     - dict[str, float]
     - Constant mixing ratios used when ``ini_mix = "const_mix"`` (e.g.
       ``{N2 = 0.78, O2 = 0.20, H2O = 1.0e-6}``). Empty by default; set
       in the sub-table ``[elements.const_mix]``.


``[atmosphere]``
----------------

Atmospheric structure: vertical grid, T–P profile, eddy diffusion,
molecular diffusion, advection, planet/star geometry.
Schema: :class:`neovulcan_config.AtmosphereConfig`.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Key
     - Type
     - Description
   * - ``atm_base``
     - str
     - Bulk gas. One of ``"H2"``, ``"N2"``, ``"O2"``, ``"CO2"``.
       Selects the binary molecular-diffusion and thermal-diffusion
       tables. Default ``"N2"``.
   * - ``rocky``
     - bool
     - ``true`` for rocky planets (surface gravity is used);
       ``false`` for gas giants. Default ``true``.
   * - ``nz``
     - int
     - **Required.** Number of vertical layers (``> 0``).
   * - ``P_b``
     - float
     - **Required.** Pressure at the bottom of the model
       (dyn cm\ :sup:`-2`, ``> 0``).
   * - ``P_t``
     - float
     - **Required.** Pressure at the top of the model
       (dyn cm\ :sup:`-2`, ``> 0``). Must satisfy ``P_t < P_b`` (the
       validator enforces this).
   * - ``use_Kzz``
     - bool
     - Include eddy diffusion. Default ``true``.
   * - ``use_moldiff``
     - bool
     - Include molecular diffusion (with thermal-diffusion drift).
       Default ``true``. Required for ``solver.ode_solver = "Rodas3"``.
   * - ``use_vm_mol``
     - bool
     - Alternative mixing-length parameterisation in place of
       molecular diffusion. Default ``false``.
   * - ``use_vz``
     - bool
     - Include vertical advection. Default ``false``.
   * - ``atm_type``
     - str
     - How :math:`T(P)` is specified: ``"isothermal"``,
       ``"analytical"``, ``"file"``, ``"vulcan_ini"``, or ``"table"``.
       Default ``"file"``.
   * - ``Kzz_prof``
     - str
     - How :math:`\Kzz(P)` is specified: ``"const"``, ``"file"``, or
       ``"Pfunc"`` (constant below ``K_p_lev`` and
       :math:`P^{-0.4}` above). Default ``"file"``.
   * - ``K_max``
     - float
     - Plateau value of :math:`\Kzz` (cm\ :sup:`2`\ s\ :sup:`-1`) used
       with ``Kzz_prof = "Pfunc"``. Default ``1.0e5``.
   * - ``K_p_lev``
     - float
     - Transition pressure (bar) where ``Kzz_prof = "Pfunc"`` switches
       from constant to :math:`P^{-0.4}` scaling. Default ``0.1``.
   * - ``K_deep``
     - float, optional
     - Lower-atmosphere plateau value used by some ``Kzz_prof``
       extensions. Omit to disable.
   * - ``vz_prof``
     - str
     - ``"const"`` or ``"file"``. Default ``"const"``.
   * - ``gs``
     - float
     - **Required.** Surface gravity (cm s\ :sup:`-2`).
   * - ``Tiso``
     - float
     - Isothermal temperature (K), only read when
       ``atm_type = "isothermal"``. Default ``1000.0``.
   * - ``para_warm``
     - list[float]
     - :math:`(T_\mathrm{int}, T_\mathrm{irr}, \kappa_L, \kappa_S,
       \beta_S, \beta_L)` parameters for the analytical T–P profile of
       Heng et al. (2014). Six elements; default
       ``[120, 1500, 0.1, 0.02, 1, 1]``.
   * - ``para_anaTP``
     - list[float], optional
     - Same six parameters used when ``atm_type = "analytical"``. If
       omitted it is aliased to ``para_warm``.
   * - ``const_Kzz``
     - float
     - Constant :math:`\Kzz` (cm\ :sup:`2`\ s\ :sup:`-1`) used when
       ``Kzz_prof = "const"``. Default ``1.0e10``.
   * - ``const_vz``
     - float
     - Constant vertical wind (cm s\ :sup:`-1`) used when
       ``vz_prof = "const"``. Default ``0.0``.
   * - ``update_frq``
     - int
     - Number of steps between updates of :math:`\mathrm{d}z` and
       :math:`\mathrm{d}z_i` due to changes in the mean molecular
       weight. Default ``100``.
   * - ``r_star``
     - float
     - Stellar radius in :math:`R_\odot`. Default ``1.0``.
   * - ``Rp``
     - float
     - **Required.** Planetary radius (cm).
   * - ``orbit_radius``
     - float
     - Planet–star distance (AU). Default ``1.0``.
   * - ``sl_angle_deg``
     - float
     - Solar zenith angle in **degrees** (the conventional dayside
       average is 58°). Default ``58``. The radian value
       ``sl_angle = math.radians(sl_angle_deg)`` is exposed as a derived
       property on the config object for convenience.
   * - ``f_diurnal``
     - float
     - Diurnal-averaging factor applied to all photolysis rates
       (0.5 for an Earth-like rotator, 1.0 for a tidally-locked planet).
       Default ``0.5``.


``[photochemistry]``
--------------------

Photochemistry and the radiative-transfer backend used to compute the
actinic flux.
Schema: :class:`neovulcan_config.PhotochemConfig`.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Key
     - Type
     - Description
   * - ``use_photo``
     - bool
     - Include photochemistry. Default ``false``.
   * - ``use_ion``
     - bool
     - Include photoionisation chemistry. Requires ``use_photo = true``
       (the validator rejects ``use_ion = true`` otherwise).
       Default ``false``.
   * - ``scat_sp``
     - list[str]
     - Bulk gases that contribute Rayleigh scattering (e.g.
       ``["H2", "He"]`` for hot Jupiters, ``["N2", "O2"]`` for Earth).
       Empty by default.
   * - ``T_cross_sp``
     - list[str]
     - Species for which temperature-dependent UV cross-sections are
       loaded. Currently available: CO\ :sub:`2`, H\ :sub:`2`\ O,
       NH\ :sub:`3`, SH, H\ :sub:`2`\ S, SO\ :sub:`2`, S\ :sub:`2`,
       COS, CS\ :sub:`2`. Start-up cost is noticeably higher when this
       list is non-empty.
   * - ``edd``
     - float
     - First Eddington coefficient :math:`\epsilon` used by the
       two-stream backend to convert the diffuse flux to actinic units.
       Default ``0.5``.
   * - ``dbin1``
     - float
     - Wavelength bin width in the VUV (nm), below ``dbin_12trans``.
       Default ``0.1``.
   * - ``dbin2``
     - float
     - Wavelength bin width in the MUV/NUV (nm), above
       ``dbin_12trans``. Default ``2.0``.
   * - ``dbin_12trans``
     - float
     - Transition wavelength (nm) where the bin width changes from
       ``dbin1`` to ``dbin2``. Default ``240.0``.
   * - ``ini_update_photo_frq``
     - int
     - Number of time steps between actinic-flux updates while the
       chemistry is still evolving rapidly. Default ``100``.
   * - ``final_update_photo_frq``
     - int
     - Number of time steps between actinic-flux updates once the
       chemistry is near steady state. Default ``5``.
   * - ``rt_scheme``
     - str
     - Radiative-transfer backend. ``"two-stream"`` (default) uses the
       delta-Eddington solver; ``"disort"`` uses the DisORT++ Python
       bindings. See :doc:`numerics`.
   * - ``surface_albedo``
     - float
     - Lambertian surface albedo at the lower boundary. Only honoured by
       the DisORT++ backend. Default ``0.0``.
   * - ``disort_nstr``
     - int
     - Number of streams (Gauss–Legendre points) used by the DisORT++
       backend. Higher = more accurate angular discretisation but
       :math:`\mathcal{O}(n_\mathrm{str}^{\,3})` more expensive per
       layer. Default ``8``.


``[boundary_conditions]``
-------------------------

Top and bottom boundary conditions.
Schema: :class:`neovulcan_config.BoundaryConfig`.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Key
     - Type
     - Description
   * - ``use_topflux``
     - bool
     - Apply a top-of-atmosphere flux from ``paths.top_BC_flux_file``.
       Default ``false``.
   * - ``use_botflux``
     - bool
     - Apply a surface flux from ``paths.bot_BC_flux_file``.
       Default ``false``.
   * - ``use_fix_sp_bot``
     - dict[str, float]
     - Fixed mixing ratios at the bottom boundary
       (e.g. ``{ H2O = 0.0143, CO2 = 4.0e-4 }``).
   * - ``diff_esc``
     - list[str]
     - Species treated with the diffusion-limited escape flux at the
       top boundary. Default ``["H"]``.
   * - ``max_flux``
     - float
     - Upper limit (cm\ :sup:`-2`\ s\ :sup:`-1`) for the escape flux.
       Default ``1.0e13``.
   * - ``use_sat_surfaceH2O``
     - bool
     - Use a surface-temperature-controlled water saturation as the
       bottom boundary for H\ :sub:`2`\ O. Default ``false``.
   * - ``use_fix_H2He``
     - bool
     - Pin the H\ :sub:`2`/He ratio at the lower boundary. Useful for
       deep hot-Jupiter columns where the H\ :sub:`2`/He inventory must
       not drift. Default ``false``.
   * - ``loss_ex``
     - list[str]
     - Species excluded from the element-conservation diagnostic
       (useful for surface sinks that legitimately consume mass).
       Default ``[]``.


``[condensation]``
------------------

Condensation, evaporation and particle settling.
Schema: :class:`neovulcan_config.CondensationConfig`.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Key
     - Type
     - Description
   * - ``use_condense``
     - bool
     - Enable condensation/evaporation reactions. Default ``false``.
   * - ``use_settling``
     - bool
     - Apply Stokes settling to condensed species. Default ``false``.
   * - ``use_relax``
     - list[str]
     - Species advanced by implicit relaxation instead of the global
       Rosenbrock step (typically ``["H2O", "H2SO4"]``). Default ``[]``.
   * - ``humidity``
     - float
     - Relative humidity for water vapour. Default ``1.0``.
   * - ``r_p``
     - dict[str, float]
     - Particle radius (cm) keyed by condensate species (e.g.
       ``{ H2O_l_s = 1.0e-2 }``). Set in sub-table
       ``[condensation.r_p]``.
   * - ``rho_p``
     - dict[str, float]
     - Particle density (g cm\ :sup:`-3`) keyed by condensate species.
       Set in sub-table ``[condensation.rho_p]``.
   * - ``start_conden_time``
     - float
     - Simulation time (s) at which condensation is switched on.
       Default ``0.0``.
   * - ``stop_conden_time``
     - float
     - Simulation time (s) after which condensation is switched off.
       Default ``5.0e8``.
   * - ``condense_sp``
     - list[str]
     - Condensable gas species. Default ``[]``.
   * - ``non_gas_sp``
     - list[str]
     - Names of the corresponding condensate species (e.g.
       ``["H2O_l_s", "H2SO4_l"]``). Default ``[]``.
   * - ``fix_species``
     - list[str]
     - Species frozen after ``fix_species_time``. Default ``[]``.
   * - ``fix_species_time``
     - float, optional
     - Simulation time (s) after which the species in ``fix_species``
       are pinned at their current values. If omitted, aliased to
       ``stop_conden_time``.
   * - ``fix_species_from_coldtrap_lev``
     - bool
     - Only pin species below the cold-trap level. Default ``true``.
   * - ``use_ini_cold_trap``
     - bool
     - Apply a cold trap to the initial state. Default ``true``.


``[solver]``
------------

ODE integrator, step-size controller, convergence diagnostics, optional
Newton finisher.
Schema: :class:`neovulcan_config.SolverConfig`.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Key
     - Type
     - Description
   * - ``ode_solver``
     - str
     - ``"Ros2"``, ``"Rodas3"`` or ``"ODESolver"`` (case-sensitive).
       Default ``"Ros2"``. ``"Rodas3"`` requires
       ``atmosphere.use_moldiff = true``.
   * - ``use_pi_controller``
     - bool
     - Use a PI step-size controller instead of the default integrating
       controller. Default ``false``.
   * - ``use_print_prog``
     - bool
     - Print a progress line every ``print_prog_num`` steps. Default
       ``true``.
   * - ``use_print_delta``
     - bool
     - Print the per-step truncation error. Default ``false``.
   * - ``print_prog_num``
     - int
     - Progress-print frequency in steps. Default ``500``.
   * - ``dttry``
     - float
     - Initial time step (s). Default ``1.0e-10``.
   * - ``trun_min``
     - float
     - Minimum total integration time (s). Default ``1.0e2``.
   * - ``runtime``
     - float
     - Maximum total integration time (s). Default ``1.0e22``.
   * - ``dt_min``
     - float
     - Hard lower bound on the time step (s). Default ``1.0e-14``.
   * - ``dt_max``
     - float, optional
     - Hard upper bound on the time step (s). If omitted, derived as
       ``runtime * 1e-5``.
   * - ``dt_var_max``
     - float
     - Maximum per-step time-step growth factor. Default ``2.0``.
   * - ``dt_var_min``
     - float
     - Minimum per-step time-step shrink factor. Default ``0.5``.
   * - ``count_min``
     - int
     - Minimum number of steps before convergence checks fire. Default
       ``120``.
   * - ``count_max``
     - int
     - Hard upper bound on the number of steps. Default ``40000``.
   * - ``atol``
     - float
     - Absolute tolerance (cm\ :sup:`-3`). Number densities below this
       are excluded from the error norms. Default ``0.1``.
   * - ``mtol``
     - float
     - Relative-tolerance floor used by the step-size controller.
       Default ``1.0e-22``.
   * - ``mtol_conv``
     - float
     - Relative-tolerance floor used by the convergence check. Default
       ``1.0e-16``.
   * - ``pos_cut``
     - float
     - Clip small positive values to zero. Default ``0.0``.
   * - ``nega_cut``
     - float
     - Threshold below which negative updates trigger a step rejection.
       Default ``-1.0``.
   * - ``loss_eps``
     - float
     - Element-conservation tolerance used by the adaptive ``rtol``
       controller. Default ``1.0e12``.
   * - ``yconv_cri``
     - float
     - Relative change :math:`\Delta\hat n` threshold (the
       :math:`\delta` in :doc:`numerics`). Default ``0.01``.
   * - ``slope_cri``
     - float
     - :math:`\Delta\hat n / \Delta\tau` threshold (the :math:`\epsilon`
       in :doc:`numerics`). Default ``1.0e-4``.
   * - ``yconv_min``
     - float
     - Relaxed alternative for ``yconv_cri``. Default ``0.1``.
   * - ``flux_cri``
     - float
     - Relative change of the actinic flux threshold. Default ``0.1``.
   * - ``flux_atol``
     - float
     - Absolute tolerance for the actinic flux
       (photons cm\ :sup:`-2`\ s\ :sup:`-1`\ nm\ :sup:`-1`). Default
       ``1.0``.
   * - ``st_factor``
     - float
     - Fraction of the integration time used for the steady-state
       window (default 0.5 means the last half is examined).
   * - ``conv_step``
     - int
     - Hard cap on the number of past steps used in the convergence
       check. Default ``500``.
   * - ``conver_ignore``
     - list[str]
     - Species excluded from the convergence diagnostic (used for
       species whose noise level dominates the dominant-:math:`\Delta`
       test but does not matter for the result). Default ``[]``.
   * - ``rtol``
     - float
     - Relative tolerance :math:`\mathcal{T}` for the step-size
       controller. Default ``1.0``.
   * - ``post_conden_rtol``
     - float
     - Value of ``rtol`` switched to after ``fix_species_time``. Default
       ``0.05``.
   * - ``use_adapt_rtol``
     - bool
     - Adapt ``rtol`` dynamically based on element-conservation drift.
       Default ``true``.
   * - ``rtol_min``
     - float
     - Lower bound for the adaptive ``rtol``. Default ``0.02``.
   * - ``rtol_max``
     - float
     - Upper bound for the adaptive ``rtol``. Default ``2.5``.
   * - ``use_newton_finisher``
     - bool
     - Enable the optional damped-Newton tail described in
       :doc:`numerics`. Default ``false``.
   * - ``newton_switch_dy``
     - float
     - Per-step fractional change below which the Newton finisher takes
       over. Default ``1.0``.
   * - ``newton_cooldown``
     - int
     - Number of Rosenbrock steps that must elapse between successive
       Newton firings. Default ``200``.
   * - ``newton_max_iter``
     - int
     - Maximum number of Newton iterations per firing. Default ``20``.
   * - ``newton_res_tol``
     - float
     - Residual norm at which the Newton finisher considers itself
       converged. Default ``1.0e-6``.
   * - ``newton_alpha_min``
     - float
     - Smallest damping factor allowed by the line-search inside the
       Newton finisher. Default ``1.0e-3``.


``[output]``
------------

Pickle file and evolution-history options.
Schema: :class:`neovulcan_config.OutputConfig`.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Key
     - Type
     - Description
   * - ``output_humanread``
     - bool
     - Also write a human-readable text dump of the final state.
       Default ``false``.
   * - ``use_shark``
     - bool
     - Easter egg.
   * - ``save_evolution``
     - bool
     - Store the abundance and time history in the output file.
       Default ``false``.
   * - ``save_evo_frq``
     - int
     - Number of steps between evolution-history samples. Default
       ``10``.
   * - ``y_time_freq``
     - int
     - Number of steps between snapshots stored in the evolution
       arrays. Default ``1``.


``[plotting]``
--------------

Live-plot, end-of-run plot, and movie options.
Schema: :class:`neovulcan_config.PlottingConfig`.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Key
     - Type
     - Description
   * - ``plot_TP``
     - bool
     - Plot the T–P profile at the start of the run. Default ``false``.
   * - ``use_live_plot``
     - bool
     - Show a live matplotlib plot of mixing-ratio profiles during
       integration. Default ``false``.
   * - ``use_live_flux``
     - bool
     - Show a live plot of the actinic flux. Default ``false``.
   * - ``use_plot_end``
     - bool
     - Plot mixing ratios after the integration finishes. Default
       ``false``.
   * - ``use_plot_evo``
     - bool
     - Plot the time evolution at each layer at the end. Default
       ``false``.
   * - ``use_save_movie``
     - bool
     - Save the live-plot frames to ``paths.movie_dir`` for later
       assembly into a movie. Default ``false``.
   * - ``use_flux_movie``
     - bool
     - Save live-flux frames. Default ``false``.
   * - ``plot_height``
     - bool
     - Use the altitude (cm) instead of pressure on the vertical axis.
       Default ``false``.
   * - ``use_PIL``
     - bool
     - Use the Pillow library to composite live plots; fall back to a
       simpler renderer if Pillow is unavailable. Default ``true``.
   * - ``live_plot_frq``
     - int
     - Number of steps between live-plot updates. Default ``10``.
   * - ``save_movie_rate``
     - int, optional
     - Number of steps between movie-frame saves. If omitted, aliased
       to ``live_plot_frq``.
   * - ``plot_spec``
     - list[str]
     - Species to include in live plots. Default ``[]``.


Cross-section validators
------------------------

A small number of consistency rules are enforced after every load and
after every programmatic override:

* ``atmosphere.P_t < atmosphere.P_b`` — the top pressure must be
  strictly smaller than the bottom one.
* ``atmosphere.para_warm`` and ``atmosphere.para_anaTP`` must each have
  exactly six elements.
* ``photochemistry.use_ion = true`` requires
  ``photochemistry.use_photo = true``.
* ``solver.ode_solver = "Rodas3"`` requires
  ``atmosphere.use_moldiff = true``.

These are all expressed as Pydantic validators in
``src/neovulcan_config.py``; their error messages identify the
offending field on failure.


Practical advice
----------------

* Start from one of the example configurations in ``cfg_examples/``
  rather than from scratch; the boundary conditions, network choice,
  and :math:`\Kzz` profile of a working setup are difficult to guess.
* For hot atmospheres (:math:`T \gtrsim 1000` K), the default
  ``elements.ini_mix = "EQ"`` together with the SNCHO photo network is
  the recommended starting point.
* For terrestrial atmospheres, set ``atmosphere.rocky = true``, choose
  ``atmosphere.atm_base`` to match the bulk gas, use
  ``elements.ini_mix = "const_mix"`` with realistic surface mixing
  ratios, and switch on condensation.
* If the solver oscillates between accepting and rejecting steps,
  reduce ``solver.atol`` (a smaller absolute tolerance is more
  permissive in this code, because trace species below ``atol`` are not
  used to judge the step size).
* If the run terminates with ``end_case != 1`` even after
  ``solver.count_max`` steps, the most common causes are (i) a poorly
  conditioned boundary condition, (ii) a missing reaction in the
  network that should close a fast cycle, or (iii) a :math:`\Kzz`
  profile with very large gradients that interact badly with the
  finite-difference stencil.
