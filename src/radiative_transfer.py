from typing import Protocol, runtime_checkable

import numpy as np

import vulcan_cfg
import chem_funs
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

        nbins = len(var.bins)
        layer_tau = np.zeros((nz, nbins))

        for sp in absp_sp:
            idx = species.index(sp)
            if sp in vulcan_cfg.T_cross_sp:
                layer_tau += var.y[:, idx, np.newaxis] * atm.dz[:, np.newaxis] * var.cross_T[sp]
            else:
                layer_tau += var.y[:, idx, np.newaxis] * atm.dz[:, np.newaxis] * var.cross[sp]

        for sp in vulcan_cfg.scat_sp:
            idx = species.index(sp)
            layer_tau += var.y[:, idx, np.newaxis] * atm.dz[:, np.newaxis] * var.cross_scat[sp]

        # cumulative optical depth from top (tau[nz]=0 boundary stays 0 from fill)
        var.tau[:-1] = np.cumsum(layer_tau[::-1], axis=0)[::-1]

    def _compute_flux(self, var, atm):
        mu_ang = -1. * np.cos(vulcan_cfg.sl_angle)
        edd = vulcan_cfg.edd
        tau = var.tau

        delta_tau = tau[:-1] - tau[1:]

        nbins = len(var.bins)
        tot_abs  = np.zeros((nz, nbins))
        tot_scat = np.zeros((nz, nbins))
        for sp in var.photo_sp:
            tot_abs += var.ymix[:, species.index(sp), np.newaxis] * var.cross[sp]
        for sp in vulcan_cfg.scat_sp:
            tot_scat += var.ymix[:, species.index(sp), np.newaxis] * var.cross_scat[sp]

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

    def _compute_J(self, var, atm):
        flux         = var.aflux
        diss_cross   = var.cross_J
        diss_cross_T = var.cross_J_T
        n_branch     = var.n_branch
        idx          = var.sflux_din12_indx

        var.J_sp = {(sp, bn): np.zeros(nz)
                    for sp in var.photo_sp
                    for bn in range(n_branch[sp] + 1)}

        for sp in var.photo_sp:
            for nbr in range(1, n_branch[sp] + 1):
                if sp in vulcan_cfg.T_cross_sp:
                    cross = diss_cross_T[(sp, nbr)]
                else:
                    cross = diss_cross[(sp, nbr)][np.newaxis]
                val = self._spectral_integral(flux, cross, idx, var.dbin1, var.dbin2)

                var.J_sp[(sp, nbr)]  = val
                var.J_sp[(sp, 0)]   += val
                if var.pho_rate_index[(sp, nbr)] not in vulcan_cfg.remove_list:
                    var.k[var.pho_rate_index[(sp, nbr)]] = val * vulcan_cfg.f_diurnal

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
