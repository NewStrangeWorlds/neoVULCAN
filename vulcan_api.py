import os
import sys
import time
import subprocess

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _regen_chemistry_jax(base_dir, config_path):
    """Run make_chemistry_jax.py from base_dir so relative paths inside it work."""
    subprocess.run(
        [sys.executable, os.path.join(base_dir, 'src', 'make_chemistry_jax.py'),
         '-c', config_path],
        cwd=base_dir,
        check=True,
    )


def _purge_vulcan_modules():
    """Remove cached src module imports so they re-execute against a fresh config singleton."""
    for mod in [
        'store', 'build_atm', 'chemistry_jax', 'rates', 'integration',
        'ros2', 'rodas3', 'ode_solver', 'output', 'condensation', 'radiative_transfer',
        'phy_const', 'vulcan_cfg',
    ]:
        sys.modules.pop(mod, None)


class _NullOutput:
    """Drop-in replacement for Output that suppresses all file I/O and plotting."""

    def save_cfg(self, *a, **kw): pass
    def save_out(self, *a, **kw): pass
    def print_prog(self, *a, **kw): pass
    def print_end_msg(self, *a, **kw): pass
    def print_unconverged_msg(self, *a, **kw): pass
    def plot_TP(self, *a, **kw): pass
    def plot_update(self, *a, **kw): pass
    def plot_flux_update(self, *a, **kw): pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class VulcanChemistry:
    """
    Library interface to neoVULCAN for use in external atmospheric models.

    Typical usage::

        import sys
        sys.path.insert(0, '/path/to/VULCAN')
        from neoVULCAN import VulcanChemistry

        BASE = '/path/to/neoVULCAN'
        chem = VulcanChemistry(BASE, cfg_overrides={'use_photo': True})
        chem.initialize()

        for step in range(n_steps):
            T_new, P_new, Kzz_new = atm_model.get_profiles()
            chem.set_atmosphere(T=T_new, P=P_new, Kzz=Kzz_new)
            chem.run_to_convergence()
            ymix = chem.get_mixing_ratios()        # shape (nz, ni)
            atm_model.update_chemistry(ymix, chem.species)

    Notes
    -----
    Two instances with different ``nz`` or different chemical networks cannot
    coexist in the same Python process because ``chemistry_jax.py`` is a
    generated file with those values baked in.
    """

    def __init__(self, base_dir, config_path=None, cfg_overrides=None):
        """
        Parameters
        ----------
        base_dir : str
            Path to the neoVULCAN directory (contains vulcan.py, vulcan_cfg.toml).
        config_path : str, optional
            Path to the TOML config. Default: ``<base_dir>/vulcan_cfg.toml``.
        cfg_overrides : dict, optional
            Nested dict of overrides, e.g. ``{'solver': {'rtol': 0.5}}``. Merged
            into the loaded TOML before validation.
        """
        self.base_dir = os.path.abspath(base_dir)
        self.config_path = config_path or os.path.join(self.base_dir, 'vulcan_cfg.toml')
        self._cfg_overrides = cfg_overrides or {}
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, regenerate_chemistry=True):
        """
        Full one-time setup: build the atmospheric grid, load the reaction
        network and rate coefficients, and compute initial chemical abundances.

        Parameters
        ----------
        regenerate_chemistry : bool
            If True (default), re-run ``make_chemistry_jax.py`` to rebuild the
            auto-generated chemistry module from the network file. Pass False
            to skip this if the network has not changed (faster startup).
        """
        base = self.base_dir

        if regenerate_chemistry:
            _regen_chemistry_jax(base, self.config_path)

        # Ensure src/ and the VULCAN root are importable
        src = os.path.join(base, 'src')
        for p in (src, base):
            if p not in sys.path:
                sys.path.insert(0, p)

        # Purge any cached src/* modules so they re-execute against our config.
        _purge_vulcan_modules()

        # Load TOML and install as the process singleton before any src/ import.
        # The vulcan_cfg shim and src/ modules both read from this singleton.
        from neovulcan_config import VulcanConfig
        from neovulcan_runtime import set_cfg
        self._cfg = VulcanConfig.from_toml(
            self.config_path, base_dir=base, overrides=self._cfg_overrides,
            absolutify_paths=True,
        )
        set_cfg(self._cfg)
        self._initial_rtol = self._cfg.solver.rtol

        import store
        import build_atm
        import chemistry_jax as chem_funs
        from rates import ReadRate
        from integration import Integration
        from ros2 import Ros2
        from ode_solver import ODESolver
        from phy_const import kb

        # Cache references needed by other methods
        self._store = store
        self._build_atm = build_atm
        self._ReadRate = ReadRate
        self._Integration = Integration
        self._Ros2 = Ros2
        self._ODESolver = ODESolver
        self._kb = kb
        self.species = list(chem_funs.spec_list)

        null = _NullOutput()

        # --- Data containers ---
        self.data_var = store.Variables()
        self.data_atm = store.AtmData()
        self.data_para = store.Parameters()
        self.data_para.start_time = time.time()

        # --- Atmospheric grid and T-P-Kzz profile ---
        self._make_atm = build_atm.Atm()
        self.data_atm = self._make_atm.f_pico(self.data_atm)
        self.data_atm = self._make_atm.load_TPK(self.data_atm)
        if self._cfg.condensation.use_condense:
            self._make_atm.sp_sat(self.data_atm)

        # --- Reaction network and rate coefficients ---
        self._rate = ReadRate()
        self.data_var = self._rate.read_rate(self.data_var, self.data_atm)
        if self._cfg.network.use_lowT_limit_rates:
            self.data_var = self._rate.lim_lowT_rates(self.data_var, self.data_atm)
        self.data_var = self._rate.rev_rate(self.data_var, self.data_atm)
        self.data_var = self._rate.remove_rate(self.data_var)

        # --- Initial chemical abundances ---
        ini_abun = build_atm.InitialAbun()
        self._ini_abun = ini_abun
        self.data_var = ini_abun.ini_y(self.data_var, self.data_atm)
        self.data_var = ini_abun.ele_sum(self.data_var)

        # --- Grid geometry, molecular diffusion, boundary conditions ---
        self.data_atm = self._make_atm.f_mu_dz(self.data_var, self.data_atm, null)
        self._make_atm.mol_diff(self.data_atm)
        self._make_atm.BC_flux(self.data_atm)

        # --- ODE solver ---
        _solvers = {'Ros2': Ros2, 'ODESolver': ODESolver}
        self._solver = _solvers[self._cfg.solver.ode_solver]()

        # --- Photochemistry setup ---
        if self._cfg.photochemistry.use_photo:
            self._rate.make_bins_read_cross(self.data_var, self.data_atm)
            self._make_atm.read_sflux(self.data_var, self.data_atm)
            self._solver.rt(self.data_var, self.data_atm)
            self.data_var = self._rate.remove_rate(self.data_var)

        self._integ = Integration(self._solver, null)
        self._solver.naming_solver(self.data_para)

        self._initialized = True
        self._has_run = False

    # ------------------------------------------------------------------
    # Per-step interface
    # ------------------------------------------------------------------

    def set_atmosphere(self, T=None, P=None, Kzz=None):
        """
        Update the atmospheric profile and recompute all derived quantities.

        Call this before ``run_to_convergence()`` whenever the external model
        advances its own time step and provides updated conditions.

        Parameters
        ----------
        T   : array_like of shape (nz,), temperature in K
        P   : array_like of shape (nz,), pressure in dyne/cm²
        Kzz : array_like of shape (nz-1,), eddy diffusion coefficient in cm²/s
        """
        null = _NullOutput()

        if T is not None:
            self.data_atm.Tco = np.asarray(T, dtype=float)
        if P is not None:
            self.data_atm.pco = np.asarray(P, dtype=float)
            self.data_atm = self._make_atm.f_pico(self.data_atm)
        if Kzz is not None:
            self.data_atm.Kzz = np.asarray(Kzz, dtype=float)

        # Recompute total number density from updated P and T
        self.data_atm.M = self.data_atm.pco / (self._kb * self.data_atm.Tco)
        self.data_atm.n_0 = self.data_atm.M.copy()

        # Keep mixing ratios unchanged; update number densities for new n_0
        self.data_var.y = self.data_atm.n_0[:, np.newaxis] * self.data_var.ymix

        # Recompute grid geometry (dz, dzi, Hp, ...) and molecular diffusion
        self.data_atm = self._make_atm.f_mu_dz(self.data_var, self.data_atm, null)
        self._make_atm.mol_diff(self.data_atm)

        # Recompute thermal rate coefficients for the new T profile
        self.data_var = self._rate.read_rate(self.data_var, self.data_atm)
        if self._cfg.network.use_lowT_limit_rates:
            self.data_var = self._rate.lim_lowT_rates(self.data_var, self.data_atm)
        self.data_var = self._rate.rev_rate(self.data_var, self.data_atm)
        self.data_var = self._rate.remove_rate(self.data_var)

        if self._cfg.photochemistry.use_photo:
            self._solver.rt(self.data_var, self.data_atm)

    def run_to_convergence(self, warm_start=False, dt_factor=1e-3, count_min=20):
        """
        Run the chemistry integration loop until VULCAN's steady-state
        convergence criterion is satisfied (or runtime/count_max is exceeded).

        Starts from the current chemical state, so consecutive calls converge
        faster as the atmosphere approaches steady state.

        Parameters
        ----------
        warm_start : bool
            Coupled-mode restart: the composition is the steady state of a
            slightly different atmosphere (the previous pass of an
            atmosphere/chemistry coupling loop). Instead of restarting the time
            step from ``dttry`` (1e-10 s by default) and taking at least
            ``solver.count_min`` steps, the previous time step scaled by
            ``dt_factor`` is kept (floored at ``dttry``) and the minimum step
            count is lowered to ``count_min``. The step controller shrinks the
            step further on its own if the change was larger than expected.
            Ignored on the first call.
        dt_factor : float
            Fraction of the previous time step to restart from (warm_start only).
        count_min : int
            Minimum number of steps before convergence may be declared
            (warm_start only).

        Notes
        -----
        Every call resets the integration clock: VULCAN's convergence criterion
        (``longdy``) measures the relative change over the last ``st_factor``
        fraction of the *current* run, so it needs a fresh history.
        """
        store = self._store
        cfg_solver = self._cfg.solver

        warm = warm_start and self._has_run
        dt_start = max(cfg_solver.dttry, self.data_var.dt * dt_factor) if warm else cfg_solver.dttry

        # Fresh counters and timing — but keep the current chemistry in data_var.y.
        # The solver's dispatch name lives on the parameter object, so it must be re-stamped
        # (previously it was stamped only on the object created by initialize(), and every
        # run_to_convergence() call dispatched on an empty name).
        self.data_para = store.Parameters()
        self.data_para.start_time = time.time()
        self._solver.naming_solver(self.data_para)

        # Reset integration-time variables so convergence checks work correctly
        self.data_var.t = 0.
        self.data_var.dt = dt_start
        self.data_var.y_time = []
        self.data_var.t_time = []
        self.data_var.atom_loss_time = []
        self.data_var.longdy = 1.
        self.data_var.longdydt = 1.
        self.data_var.dy = 1.
        self.data_var.dy_prev = 1.
        self.data_var.aflux_change = 0.

        # Restore rtol to its configured starting value (may have been adapted)
        self._cfg.solver.rtol = self._initial_rtol

        # Reset photo-update frequency in case it was switched to final value
        if self._cfg.photochemistry.use_photo:
            self._integ.update_photo_frq = self._cfg.photochemistry.ini_update_photo_frq

        saved_count_min = cfg_solver.count_min
        if warm:
            cfg_solver.count_min = count_min
        try:
            self._integ(self.data_var, self.data_atm, self.data_para, self._make_atm)
        finally:
            cfg_solver.count_min = saved_count_min
        self._has_run = True

    def set_composition(self, species, mixing_ratios, renormalize=True):
        """
        Overwrite the current mixing ratios of the given species.

        Used to seed the kinetics from an atmosphere model's composition (an
        equilibrium or quench-approximation profile) instead of neoVULCAN's own
        FastChem initial state, or to hand back a relaxed composition in a
        coupling loop.

        Parameters
        ----------
        species : list of str
            Species names as in ``self.species``; unknown names are ignored.
        mixing_ratios : array_like of shape (nz, len(species))
            Volume mixing ratios on neoVULCAN's own grid, bottom to top.
        renormalize : bool
            Rescale every level so the mixing ratios sum to one afterwards
            (species not in ``species`` keep their current values).

        Notes
        -----
        The elemental budget the integration conserves is re-referenced to the
        new composition, so atom-loss diagnostics (which also drive the adaptive
        tolerance) measure drift from this seed rather than from the original
        FastChem state.
        """
        table = np.asarray(mixing_ratios, dtype=float)
        ymix = self.data_var.ymix
        if table.shape != (ymix.shape[0], len(species)):
            raise ValueError(
                f'mixing_ratios must have shape ({ymix.shape[0]}, {len(species)}), got {table.shape}')

        known = [(k, self.species.index(sp)) for k, sp in enumerate(species) if sp in self.species]
        ignored = [sp for sp in species if sp not in self.species]
        if not known:
            raise ValueError('none of the supplied species is in the network')

        for k, j in known:
            ymix[:, j] = np.maximum(table[:, k], 0.0)

        if renormalize:
            ymix /= ymix.sum(axis=1)[:, np.newaxis]

        self.data_var.ymix = ymix
        self.data_var.y = self.data_atm.n_0[:, np.newaxis] * ymix
        self.data_var = self._ini_abun.ele_sum(self.data_var)
        return ignored

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    def get_mixing_ratios(self):
        """Return mixing ratios as a copy, shape ``(nz, ni)``."""
        return np.copy(self.data_var.ymix)

    def get_number_densities(self):
        """Return number densities in cm⁻³ as a copy, shape ``(nz, ni)``."""
        return np.copy(self.data_var.y)

    def get_convergence_info(self):
        """
        Return a dict describing the outcome of the last ``run_to_convergence()`` call.

        Keys
        ----
        end_case : int
            1 = converged, 2 = runtime exceeded, 3 = step count exceeded.
        longdy : float
            Long-term fractional change in mixing ratios (convergence metric).
        longdydt : float
            Time derivative of longdy (slope convergence metric).
        count : int
            Number of ODE steps taken.
        """
        return {
            'end_case': self.data_para.end_case,
            'longdy': self.data_var.longdy,
            'longdydt': self.data_var.longdydt,
            'count': self.data_para.count,
        }
