from typing import Protocol, runtime_checkable

import numpy as np

import vulcan_cfg
import chemistry_jax as chem_funs
from phy_const import hc, ag0
from vulcan_cfg import nz

species = chem_funs.spec_list


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
        
        if vulcan_cfg.use_ion:
            self._compute_Jion(var, atm)

    def _compute_tau(self, var, atm):
        var.tau.fill(0)
        absp_sp = sorted(set(var.photo_sp) | set(var.ion_sp))

        # Cache stacked cross-section arrays for the matmul.  Split into
        # constant-cross species (cross_sp shape (nbins,)) and T-dependent
        # species (cross_T_sp shape (nz, nbins)) which can't be stacked the
        # same way.
        tau_key = (tuple(absp_sp), tuple(sorted(vulcan_cfg.T_cross_sp)),
                   tuple(vulcan_cfg.scat_sp))
        if getattr(self, '_tau_cache_key', None) != tau_key:
            sp_const = [sp for sp in absp_sp if sp not in vulcan_cfg.T_cross_sp]
            sp_T     = [sp for sp in absp_sp if sp in vulcan_cfg.T_cross_sp]
            self._tau_const_idx   = np.array([species.index(sp) for sp in sp_const])
            self._tau_const_cross = (np.stack([var.cross[sp] for sp in sp_const])
                                     if sp_const else None)         # (n_c, nbins)
            self._tau_T_specs     = [(species.index(sp), sp) for sp in sp_T]
            self._tau_scat_idx    = np.array([species.index(sp) for sp in vulcan_cfg.scat_sp])
            self._tau_scat_cross  = (np.stack([var.cross_scat[sp] for sp in vulcan_cfg.scat_sp])
                                     if vulcan_cfg.scat_sp else None)
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
            self._scat_idx         = np.array([species.index(sp) for sp in vulcan_cfg.scat_sp])
            self._scat_cross_stk   = np.stack([var.cross_scat[sp] for sp in vulcan_cfg.scat_sp])
            self._photo_cache_key  = photo_sp_key

        mu_ang = -1. * np.cos(vulcan_cfg.sl_angle)
        edd = vulcan_cfg.edd
        tau = var.tau

        delta_tau = tau[:-1] - tau[1:]

        # Species accumulation as two matmuls instead of two Python loops.
        tot_abs  = var.ymix[:, self._photo_idx] @ self._photo_cross_stk   # (nz, nbins)
        tot_scat = var.ymix[:, self._scat_idx]  @ self._scat_cross_stk

        w0 = tot_scat / (tot_abs + tot_scat)
        w0 = np.nan_to_num(w0)
        w0 = np.minimum(w0, 1. - 1.e-8)

        var.sflux = var.sflux_top * np.exp(-1. * tau / np.cos(vulcan_cfg.sl_angle))
        dir_flux  = var.sflux * np.cos(vulcan_cfg.sl_angle)

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

        for j in range(nz-1, -1, -1):
            var.dflux_d[j] = 1./chi[j] * (phi[j]*var.dflux_d[j+1] - xi[j]*var.dflux_u[j] + i_d[j]/mu_ang)
        for j in range(1, nz+1):
            var.dflux_u[j] = 1./chi[j-1] * (phi[j-1]*var.dflux_u[j-1] - xi[j-1]*var.dflux_d[j] + i_u[j-1]/mu_ang)

        ave_dir_flux = 0.5 * (var.sflux[:-1] + var.sflux[1:])
        tot_flux = (ave_dir_flux
                    + 0.5*(var.dflux_u[:-1] + var.dflux_u[1:]
                           + var.dflux_d[1:] + var.dflux_d[:-1]) / edd)

        var.prev_aflux = np.copy(var.aflux)
        var.aflux = tot_flux / (hc / var.bins)
        var.aflux_change = np.nanmax(
            np.abs(var.aflux - var.prev_aflux)[var.aflux > vulcan_cfg.flux_atol]
            / var.aflux[var.aflux > vulcan_cfg.flux_atol]
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
                 tuple(sorted(vulcan_cfg.T_cross_sp)),
                 int(idx), var.dbin1, var.dbin2)
        if getattr(self, '_J_cache_key', None) != j_key:
            nbins   = len(var.bins)
            weights = self._trapezoidal_weights(nbins, idx, var.dbin1, var.dbin2)
            const_keys, T_keys = [], []
            for sp in sorted(var.photo_sp, key=species.index):
                for nbr in range(1, n_branch[sp] + 1):
                    if sp in vulcan_cfg.T_cross_sp:
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

        f_diurnal   = vulcan_cfg.f_diurnal
        remove_list = vulcan_cfg.remove_list
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
                if var.ion_rate_index[(sp, nbr)] not in vulcan_cfg.remove_list:
                    var.k[var.ion_rate_index[(sp, nbr)]] = val * vulcan_cfg.f_diurnal
