from typing import Protocol, runtime_checkable

import numpy as np

from neovulcan_runtime import get_cfg
cfg = get_cfg()
import chemistry_jax as chem_funs
from phy_const import hc, ag0
nz = cfg.atmosphere.nz

species = chem_funs.spec_list


def make_rt():
    """Construct the radiative-transfer solver selected by cfg.photochemistry.rt_scheme."""
    scheme = cfg.photochemistry.rt_scheme.lower()
    if scheme in ('two-stream', 'twostream', '2-stream', '2stream'):
        return TwoStreamRT()
    if scheme in ('disort', 'disortpp', 'disort++'):
        return DisortRT()
    raise ValueError(f"Unknown rt_scheme {scheme!r}; expected 'two-stream' or 'disort'.")


@runtime_checkable
class RadiativeTransfer(Protocol):
    """Protocol for radiative transfer schemes.

    A conforming implementation is callable as rt(var, atm) and updates
    var.k with photodissociation and photoionisation rate coefficients.
    """
    def __call__(self, var, atm) -> None: ...


class TwoStreamRT:
    """Eddington two-stream (delta-Eddington) radiative transfer.

    Propagates a direct stellar beam plus diffuse up/down fluxes layer-by-layer
    to produce the actinic flux, then integrates cross-sections to obtain
    photodissociation (J_sp) and photoionisation (Jion_sp) rate coefficients
    written back into var.k.
    """

    def __call__(self, var, atm) -> None:
        """Run the full RT pipeline, updating var.k with J and Jion rates."""
        self._compute_tau(var, atm)
        self._compute_flux(var, atm)
        self._compute_J(var, atm)
        
        if cfg.photochemistry.use_ion:
            self._compute_Jion(var, atm)

    def _compute_tau(self, var, atm):
        var.tau.fill(0)
        absp_sp = sorted(set(var.photo_sp) | set(var.ion_sp))

        # Cache stacked cross-section arrays for the matmul.  Split into
        # constant-cross species (cross_sp shape (nbins,)) and T-dependent
        # species (cross_T_sp shape (nz, nbins)) which can't be stacked the
        # same way.
        tau_key = (tuple(absp_sp), tuple(sorted(cfg.photochemistry.T_cross_sp)),
                   tuple(cfg.photochemistry.scat_sp))
        if getattr(self, '_tau_cache_key', None) != tau_key:
            sp_const = [sp for sp in absp_sp if sp not in cfg.photochemistry.T_cross_sp]
            sp_T     = [sp for sp in absp_sp if sp in cfg.photochemistry.T_cross_sp]
            self._tau_const_idx   = np.array([species.index(sp) for sp in sp_const])
            self._tau_const_cross = (np.stack([var.cross[sp] for sp in sp_const])
                                     if sp_const else None)         # (n_c, nbins)
            self._tau_T_specs     = [(species.index(sp), sp) for sp in sp_T]
            self._tau_scat_idx    = np.array([species.index(sp) for sp in cfg.photochemistry.scat_sp])
            self._tau_scat_cross  = (np.stack([var.cross_scat[sp] for sp in cfg.photochemistry.scat_sp])
                                     if cfg.photochemistry.scat_sp else None)
            self._tau_cache_key   = tau_key

        # Constant-cross absorbers + scattering — one matmul each.
        if self._tau_const_cross is not None:
            layer_tau = atm.dz[:, None] * (var.y[:, self._tau_const_idx] @ self._tau_const_cross)
        else:
            layer_tau = np.zeros((nz, len(var.bins)))
        if self._tau_scat_cross is not None:
            layer_tau += atm.dz[:, None] * (var.y[:, self._tau_scat_idx] @ self._tau_scat_cross)

        # T-dependent absorbers: cross is per-layer per-species; loop kept.
        for idx, sp in self._tau_T_specs:
            layer_tau += var.y[:, idx, np.newaxis] * atm.dz[:, np.newaxis] * var.cross_T[sp]

        # cumulative optical depth from top (tau[nz]=0 boundary stays 0 from fill)
        var.tau[:-1] = np.cumsum(layer_tau[::-1], axis=0)[::-1]

    def _compute_flux(self, var, atm):
        # Stacked cross-section arrays for the species-loop matmul.  Rebuilt
        # only if the species list changes (effectively once per run).  Key
        # on the sorted species-name tuple so rebuilding `var` (e.g. in
        # tests) doesn't invalidate the cache.
        photo_sp_key = tuple(sorted(var.photo_sp))
        if getattr(self, '_photo_cache_key', None) != photo_sp_key:
            sp_sorted = sorted(var.photo_sp, key=species.index)
            self._photo_idx        = np.array([species.index(sp) for sp in sp_sorted])
            self._photo_cross_stk  = np.stack([var.cross[sp] for sp in sp_sorted])  # (n_p, nbins)
            self._scat_idx         = np.array([species.index(sp) for sp in cfg.photochemistry.scat_sp])
            self._scat_cross_stk   = np.stack([var.cross_scat[sp] for sp in cfg.photochemistry.scat_sp])
            self._photo_cache_key  = photo_sp_key

        mu_ang = -1. * np.cos(cfg.atmosphere.sl_angle)
        edd = cfg.photochemistry.edd
        tau = var.tau

        delta_tau = tau[:-1] - tau[1:]

        # Species accumulation as two matmuls instead of two Python loops.
        tot_abs  = var.ymix[:, self._photo_idx] @ self._photo_cross_stk   # (nz, nbins)
        tot_scat = var.ymix[:, self._scat_idx]  @ self._scat_cross_stk

        w0 = tot_scat / (tot_abs + tot_scat)
        w0 = np.nan_to_num(w0)
        w0 = np.minimum(w0, 1. - 1.e-8)

        var.sflux = var.sflux_top * np.exp(-1. * tau / np.cos(cfg.atmosphere.sl_angle))
        dir_flux  = var.sflux * np.cos(cfg.atmosphere.sl_angle)

        if ag0 == 0:
            tran   = np.exp(-1./edd * (1. - w0)**0.5 * delta_tau)
            zeta_p = 0.5 * (1. + (1. - w0)**0.5)
            zeta_m = 0.5 * (1. - (1. - w0)**0.5)
            ll     = -1. * w0 / (1./mu_ang**2 - 1./edd**2 * (1. - w0))
            g_p    = 0.5 * (ll * (1./edd + 1./mu_ang))
            g_m    = 0.5 * (ll * (1./edd - 1./mu_ang))
        else:
            tran   = np.exp(-1./edd * ((1. - w0*ag0) * (1. - w0))**0.5 * delta_tau)
            zeta_p = 0.5 * (1. + ((1. - w0) / (1 - w0*ag0))**0.5)
            zeta_m = 0.5 * (1. - ((1. - w0) / (1 - w0*ag0))**0.5)
            ll     = ((1. - w0) * (1 - w0*ag0) - 1.) / (1./mu_ang**2 - 1./edd**2 * (1. - w0) * (1 - w0*ag0))
            g_p    = 0.5 * (ll * (1./edd + 1./(mu_ang*(1. - w0*ag0))) + w0*ag0*mu_ang/(1. - w0*ag0))
            g_m    = 0.5 * (ll * (1./edd - 1./(mu_ang*(1. - w0*ag0))) - w0*ag0*mu_ang/(1. - w0*ag0))

        ll = np.minimum(ll,  1.e10)
        ll = np.maximum(ll, -1.e10)

        chi = zeta_m**2 * tran**2 - zeta_p**2
        xi  = zeta_p * zeta_m * (1. - tran**2)
        phi = (zeta_m**2 - zeta_p**2) * tran

        i_u = phi*g_p*dir_flux[:-1] - (xi*g_m  + chi*g_p) * dir_flux[1:]
        i_d = phi*g_m*dir_flux[1:]  - (chi*g_m + xi*g_p)  * dir_flux[:-1]

        var.zeta_m = zeta_m
        var.zeta_p = zeta_p
        var.tran   = tran

        # Lambertian-surface lower BC: dflux_u[0] = A · (dflux_d[0] + dir_flux[0]).
        # When A=0 (default) the sweep is single-pass exactly as before.
        # When A>0 we iterate the coupled boundary to a fixed point — contracts
        # by ≤ A per round trip, so a handful of sweeps suffices.
        sfc_alb  = float(cfg.photochemistry.surface_albedo)
        max_iter = 20 if sfc_alb > 0 else 1
        sfc_tol  = 1.e-6

        for _ in range(max_iter):
            u0_prev = var.dflux_u[0].copy() if sfc_alb > 0 else None
            for j in range(nz-1, -1, -1):
                var.dflux_d[j] = 1./chi[j] * (phi[j]*var.dflux_d[j+1] - xi[j]*var.dflux_u[j] + i_d[j]/mu_ang)
            if sfc_alb > 0:
                var.dflux_u[0] = sfc_alb * (var.dflux_d[0] + dir_flux[0])
            for j in range(1, nz+1):
                var.dflux_u[j] = 1./chi[j-1] * (phi[j-1]*var.dflux_u[j-1] - xi[j-1]*var.dflux_d[j] + i_u[j-1]/mu_ang)
            if sfc_alb == 0:
                break
            mask = var.dflux_u[0] > cfg.solver.flux_atol
            if not mask.any() or np.nanmax(
                np.abs(var.dflux_u[0][mask] - u0_prev[mask]) / var.dflux_u[0][mask]
            ) < sfc_tol:
                break

        ave_dir_flux = 0.5 * (var.sflux[:-1] + var.sflux[1:])
        tot_flux = (ave_dir_flux
                    + 0.5*(var.dflux_u[:-1] + var.dflux_u[1:]
                           + var.dflux_d[1:] + var.dflux_d[:-1]) / edd)

        var.prev_aflux = np.copy(var.aflux)
        var.aflux = tot_flux / (hc / var.bins)
        var.aflux_change = np.nanmax(
            np.abs(var.aflux - var.prev_aflux)[var.aflux > cfg.solver.flux_atol]
            / var.aflux[var.aflux > cfg.solver.flux_atol]
        )

    def _spectral_integral(self, flux, cross, idx, dbin1, dbin2):
        """Trapezoidal integration of flux * cross over two spectral regions.

        cross must have shape (nz, nbins) or (1, nbins) — 1D cross-sections
        should be passed as cross[np.newaxis] so broadcasting works uniformly.
        """
        val  = np.sum(flux[:, :idx] * cross[:, :idx] * dbin1, axis=1)
        val -= 0.5 * (flux[:, 0]     * cross[:, 0]
                    + flux[:, idx-1] * cross[:, idx-1]) * dbin1
        val += np.sum(flux[:, idx:] * cross[:, idx:] * dbin2, axis=1)
        val -= 0.5 * (flux[:, idx] * cross[:, idx]
                    + flux[:, -1]  * cross[:, -1]) * dbin2
        return val

    def _trapezoidal_weights(self, nbins, idx, dbin1, dbin2):
        """Build the per-bin weight vector that turns ``sum(w·flux·cross)``
        into the same two-region trapezoidal integral as ``_spectral_integral``.

        Region 1 (bins 0..idx-1, spacing dbin1): trapezoidal endpoints halved.
        Region 2 (bins idx..nbins-1, spacing dbin2): trapezoidal endpoints halved.
        """
        w = np.empty(nbins)
        w[:idx]  = dbin1
        w[0]     = 0.5 * dbin1
        w[idx-1] = 0.5 * dbin1
        w[idx:]  = dbin2
        w[idx]   = 0.5 * dbin2
        w[-1]    = 0.5 * dbin2
        return w

    def _compute_J(self, var, atm):
        flux         = var.aflux
        idx          = var.sflux_din12_indx
        n_branch     = var.n_branch

        # Cache stacked cross-section arrays, the trapezoidal weight vector,
        # and the flat list of (sp, nbr) keys, all keyed on inputs that only
        # change at setup.
        j_key = (tuple(sorted(var.photo_sp)),
                 tuple(sorted(cfg.photochemistry.T_cross_sp)),
                 int(idx), var.dbin1, var.dbin2)
        if getattr(self, '_J_cache_key', None) != j_key:
            nbins   = len(var.bins)
            weights = self._trapezoidal_weights(nbins, idx, var.dbin1, var.dbin2)
            const_keys, T_keys = [], []
            for sp in sorted(var.photo_sp, key=species.index):
                for nbr in range(1, n_branch[sp] + 1):
                    if sp in cfg.photochemistry.T_cross_sp:
                        T_keys.append((sp, nbr))
                    else:
                        const_keys.append((sp, nbr))
            # Stack const cross-sections AND fold weights into them so the
            # hot path is one (n_const, nbins) @ (nbins, nz) matmul.
            if const_keys:
                cross_J_w = np.stack([var.cross_J[k] for k in const_keys]) * weights
            else:
                cross_J_w = None                                   # (n_c, nbins)
            # T-dep stack: (n_T, nz, nbins); weights folded.
            if T_keys:
                cross_JT_w = np.stack([var.cross_J_T[k] for k in T_keys]) * weights
            else:
                cross_JT_w = None
            # Pre-compute the active species set for J_sp[(sp, 0)] sums.
            sp_to_branches = {}
            for k in const_keys + T_keys:
                sp_to_branches.setdefault(k[0], []).append(k)
            self._J_const_keys   = const_keys
            self._J_T_keys       = T_keys
            self._J_cross_w      = cross_J_w
            self._J_cross_T_w    = cross_JT_w
            self._J_sp_branches  = sp_to_branches
            self._J_zero_template = np.zeros(nz)
            self._J_cache_key    = j_key

        # Const-cross integrals: one matmul gives all const branches at once.
        if self._J_cross_w is not None:
            val_const = self._J_cross_w @ flux.T                  # (n_c, nz)
        else:
            val_const = np.empty((0, nz))
        # T-dep integrals: einsum 's,nb,b->s,n' fold per (species, layer).
        if self._J_cross_T_w is not None:
            val_T = np.einsum('snb,nb->sn', self._J_cross_T_w, flux)
        else:
            val_T = np.empty((0, nz))

        # Scatter into var.J_sp and var.k.  Initialise per-species sums
        # (the (sp, 0) entries) once; per-branch entries are direct
        # assignments.
        var.J_sp = {}
        for sp in var.photo_sp:
            var.J_sp[(sp, 0)] = np.zeros(nz)

        f_diurnal   = cfg.atmosphere.f_diurnal
        remove_list = cfg.network.remove_list
        pho_rate_index = var.pho_rate_index
        k_arr = var.k

        for i, (sp, nbr) in enumerate(self._J_const_keys):
            v = val_const[i]
            var.J_sp[(sp, nbr)]  = v
            var.J_sp[(sp, 0)]   += v
            ridx = pho_rate_index[(sp, nbr)]
            if ridx not in remove_list:
                k_arr[ridx] = v * f_diurnal
        for i, (sp, nbr) in enumerate(self._J_T_keys):
            v = val_T[i]
            var.J_sp[(sp, nbr)]  = v
            var.J_sp[(sp, 0)]   += v
            ridx = pho_rate_index[(sp, nbr)]
            if ridx not in remove_list:
                k_arr[ridx] = v * f_diurnal

    def _compute_Jion(self, var, atm):
        flux      = var.aflux
        ion_cross = var.cross_Jion
        n_branch  = var.ion_branch
        idx       = var.sflux_din12_indx

        var.Jion_sp = {(sp, bn): np.zeros(nz)
                       for sp in var.ion_sp
                       for bn in range(n_branch[sp] + 1)}

        for sp in var.ion_sp:
            for nbr in range(1, n_branch[sp] + 1):
                val = self._spectral_integral(
                    flux, ion_cross[(sp, nbr)][np.newaxis], idx, var.dbin1, var.dbin2
                )
                var.Jion_sp[(sp, nbr)]  = val
                var.Jion_sp[(sp, 0)]   += val
                if var.ion_rate_index[(sp, nbr)] not in cfg.network.remove_list:
                    var.k[var.ion_rate_index[(sp, nbr)]] = val * cfg.atmosphere.f_diurnal


class DisortRT(TwoStreamRT):
    """Discrete-ordinates radiative transfer using the DisORT++ Python bindings.

    Drop-in replacement for ``TwoStreamRT``: shares the optical-depth assembly
    and the J / Jion spectral integrals, but replaces the two-stream Eddington
    flux step with a per-bin DISORT solve.  Results are written into the same
    ``var`` arrays (``aflux``, ``sflux``, ``dflux_u``, ``dflux_d``) so the rest
    of the pipeline (convergence checks, plotting) is unchanged.
    """

    def __init__(self):
        super().__init__()
        import disortpp  # raise at construction if the package isn't installed
        self._disortpp = disortpp
        self._nstr     = int(cfg.photochemistry.disort_nstr)
        self._solver   = disortpp.create_flux_solver(self._nstr)
        # DisORT++ ≥ 2.2 exposes ``index_from_bottom`` on DisortFluxConfig.
        # When available, DisORT++ reverses the input/output arrays internally,
        # so we can pass VULCAN's native bottom→top ordering through untouched.
        self._native_bottom = hasattr(disortpp.DisortFluxConfig(1, self._nstr),
                                      'index_from_bottom')
        self._cfg      = None
        self._cfg_nz   = None

    def _make_cfg(self):
        """Build (and cache) the DisortFluxConfig used across bins.

        The config's per-layer arrays are mutated in the per-bin loop, but its
        size, stream count, and phase-function choice are fixed for the run.
        """
        dcfg = self._disortpp.DisortFluxConfig(nz, self._nstr)
        dcfg.direct_beam_mu = float(np.cos(cfg.atmosphere.sl_angle))
        # surface_albedo is refreshed per call in _compute_flux so changes
        # to cfg.photochemistry.surface_albedo between calls take effect.
        dcfg.surface_albedo = float(cfg.photochemistry.surface_albedo)
        if self._native_bottom:
            dcfg.index_from_bottom = True
        dcfg.allocate()
        dcfg.delta_tau          = np.zeros(nz)
        dcfg.single_scat_albedo = np.zeros(nz)
        # The bulk scatterers in cfg.photochemistry.scat_sp (e.g. N2, O2) contribute
        # Rayleigh scattering; use the matching phase function rather than the
        # 2-stream's ag0-Henyey-Greenstein approximation.
        dcfg.set_rayleigh()
        self._cfg    = dcfg
        self._cfg_nz = nz
        return dcfg

    def _compute_flux(self, var, atm):
        nbins = len(var.bins)
        mu0   = float(np.cos(cfg.atmosphere.sl_angle))

        # ------------------------------------------------------------------
        # Per-layer optical depth (bottom→top VULCAN order) reconstructed from
        # the cumulative tau just built by _compute_tau, plus a separate sum
        # of the scattering contribution for the single-scattering albedo.
        # ------------------------------------------------------------------
        layer_tau = var.tau[:-1] - var.tau[1:]                       # (nz, nbins)

        if cfg.photochemistry.scat_sp:
            scat_idx   = np.array([species.index(sp) for sp in cfg.photochemistry.scat_sp])
            scat_cross = np.stack([var.cross_scat[sp] for sp in cfg.photochemistry.scat_sp])
            layer_tau_scat = atm.dz[:, None] * (var.y[:, scat_idx] @ scat_cross)
        else:
            layer_tau_scat = np.zeros_like(layer_tau)

        with np.errstate(divide='ignore', invalid='ignore'):
            w0 = layer_tau_scat / layer_tau
        w0 = np.nan_to_num(w0, nan=0.0, posinf=0.0, neginf=0.0)
        np.minimum(w0, 1. - 1.e-8, out=w0)

        # solve_flux_spectral expects per-wavenumber × per-layer arrays; transpose
        # (nz, nbins) → (nbins, nz).  If DisORT++ supports `index_from_bottom`,
        # VULCAN's native bottom→top layer order goes through unchanged;
        # otherwise reverse the layer axis first.
        if self._native_bottom:
            delta_tau_d = layer_tau.T
            w0_d        = w0.T
        else:
            delta_tau_d = layer_tau[::-1].T
            w0_d        = w0[::-1].T

        # Single-wavenumber RT: one point per bin at the bin centre (cm^-1).
        wn = 1.e7 / var.bins

        dcfg = self._cfg if self._cfg_nz == nz else self._make_cfg()
        dcfg.surface_albedo = float(cfg.photochemistry.surface_albedo)

        # ------------------------------------------------------------------
        # Single batch call — disortpp runs the per-bin loop in C++ with OpenMP.
        # Per-wavenumber overrides supply the layer optical properties and the
        # direct-beam magnitude; everything else (μ₀, phase function, surface
        # albedo) stays on the shared config.
        # ------------------------------------------------------------------
        results = self._disortpp.solve_flux_spectral(
            dcfg, wn,
            delta_tau=delta_tau_d,
            single_scat_albedo=w0_d,
            direct_beam_flux=var.sflux_top,
        )

        var.prev_aflux = np.copy(var.aflux)
        sflux   = np.empty_like(var.sflux)
        dflux_u = np.empty_like(var.dflux_u)
        dflux_d = np.empty_like(var.dflux_d)
        tot_flux = np.empty((nz, nbins))
        four_pi = 4. * np.pi
        native  = self._native_bottom
        inv_mu0 = 1.0 / mu0 if mu0 > 0 else 1.0

        for b, r in enumerate(results):
            mi   = np.asarray(r.mean_intensity)
            fdir = np.asarray(r.flux_direct_beam)
            fup  = np.asarray(r.flux_up)
            fdn  = np.asarray(r.flux_down)
            if not native:
                mi, fdir, fup, fdn = mi[::-1], fdir[::-1], fup[::-1], fdn[::-1]

            actinic_lev = four_pi * mi
            tot_flux[:, b] = 0.5 * (actinic_lev[:-1] + actinic_lev[1:])
            sflux[:, b]    = fdir * inv_mu0
            dflux_u[:, b]  = fup
            dflux_d[:, b]  = fdn

        var.sflux   = sflux
        var.dflux_u = dflux_u
        var.dflux_d = dflux_d

        # Energy → photon flux (photons / cm^2 / s / nm); same convention as 2-stream.
        var.aflux = tot_flux / (hc / var.bins)
        mask = var.aflux > cfg.solver.flux_atol
        if mask.any():
            var.aflux_change = float(np.nanmax(
                np.abs(var.aflux - var.prev_aflux)[mask] / var.aflux[mask]
            ))
        else:
            var.aflux_change = 0.0
