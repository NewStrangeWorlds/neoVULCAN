import numpy as np

import vulcan_cfg
import chem_funs
from phy_const import kb, Navo
from vulcan_cfg import nz

species = chem_funs.spec_list


class Condensation:
    """Condensation and evaporation rate updates in the continuum diffusion regime."""

    # reaction string -> (gas_sp, conden_sp, m, has_relax, has_humidity)
    # has_relax:    rates are zeroed when vulcan_cfg.use_relax is set (relax methods take over)
    # has_humidity: saturation density is multiplied by vulcan_cfg.humidity
    _CONDEN_PARAMS = {
        'H2O -> H2O_l_s':   ('H2O',   'H2O_l_s', 18.    /Navo, True,  True ),
        'NH3 -> NH3_l':     ('NH3',   'NH3_l_s', 17.    /Navo, True,  False),
        'H2SO4 -> H2SO4_l': ('H2SO4', 'H2SO4_l', 98.022 /Navo, False, False),
        'S2 -> S2_l_s':     ('S2',    'S2_l_s',  45.019 /Navo, False, False),
        'S4 -> S4_l_s':     ('S4',    'S4_l_s',  32.06*4/Navo, False, False),
        'S8 -> S8_l_s':     ('S8',    'S8_l_s',  360.152/Navo, False, False),
        'C -> C_s':         ('C',     'C_s',     12.011 /Navo, False, False),
    }

    def update(self, var, atm):
        """Update condensation reaction rate coefficients var.k for the current number densities."""
        for re in var.conden_re_list:
            rf = var.Rf[re]
            if rf not in self._CONDEN_PARAMS:
                continue
            gas_sp, conden_sp, m, has_relax, has_humidity = self._CONDEN_PARAMS[rf]
            if gas_sp not in vulcan_cfg.condense_sp:
                continue

            if has_relax and vulcan_cfg.use_relax:
                var.k[re]   = np.repeat(0., nz)
                var.k[re+1] = np.repeat(0., nz)
                continue

            rho_p = atm.rho_p[conden_sp]
            r_p   = atm.r_p[conden_sp]
            sat   = atm.sat_p[gas_sp]/kb/atm.Tco
            if has_humidity:
                sat *= vulcan_cfg.humidity

            Dg   = np.insert(atm.Dzz[:, species.index(gas_sp)], 0, atm.Dzz[0, species.index(gas_sp)])
            rate = Dg * m/rho_p /r_p**2 * (var.y[:, species.index(gas_sp)] - sat)

            var.k[re]   = np.maximum(rate, 0)
            var.k[re+1] = np.abs(np.minimum(rate, 0))

        return var

    def h2o_evap_relax(self, var, atm):
        m     = 18./Navo
        rho_p = atm.rho_p['H2O_l_s']
        r_p   = atm.r_p['H2O_l_s']
        sat_humidity = atm.sat_p['H2O']/kb/atm.Tco * vulcan_cfg.humidity

        Dg  = np.insert(atm.Dzz[:, species.index('H2O')], 0, atm.Dzz[0, species.index('H2O')])
        tau = 1./(Dg * m/rho_p /r_p**2 * (var.y[:, species.index('H2O')] - sat_humidity))
        conden_indx = np.where(tau > 0)
        evap_indx   = np.where(tau < 0)
        sat_mix = sat_humidity/atm.n_0

        # implicit-Euler condensation
        y_conden = (var.ymix[:, species.index('H2O')] + var.dt/tau*sat_mix) / (1. + var.dt/tau)

        # evaporation: both tau < 0 and (y_H2O - sat) < 0, so ice_loss > 0
        ice_loss = (var.y[:, species.index('H2O')] - sat_humidity)*var.dt/tau
        ice_loss = np.minimum(var.y[:, species.index('H2O_l_s')], ice_loss)

        var.ymix[conden_indx, species.index('H2O_l_s')] += (var.ymix[conden_indx, species.index('H2O')] - y_conden[conden_indx])
        var.ymix[conden_indx, species.index('H2O')]     = y_conden[conden_indx]

        var.ymix[evap_indx, species.index('H2O')]     += ice_loss[evap_indx]/atm.n_0[evap_indx]
        var.ymix[evap_indx, species.index('H2O_l_s')] -= ice_loss[evap_indx]/atm.n_0[evap_indx]

        var.y = var.ymix * np.vstack(np.sum(var.y[:, atm.gas_indx], axis=1))
        return var

    def nh3_evap_relax(self, var, atm):
        m     = 17./Navo
        rho_p = atm.rho_p['NH3_l_s']
        r_p   = atm.r_p['NH3_l_s']
        sat_p   = atm.sat_p['NH3']/kb/atm.Tco
        sat_mix = sat_p/atm.n_0

        conden_top = np.argmin(sat_mix)

        Dg  = np.insert(atm.Dzz[:, species.index('NH3')], 0, atm.Dzz[0, species.index('NH3')])
        tau = 1./(Dg * m/rho_p /r_p**2 * (var.y[:, species.index('NH3')] - sat_p))
        conden_indx = np.where(tau > 0)[0]
        evap_indx   = np.where(tau < 0)[0]

        # above the condensation-zone top, no condensation when using the relaxation method
        conden_indx = [i for i in conden_indx if i <= conden_top]

        # implicit-Euler condensation
        y_conden = (var.ymix[:, species.index('NH3')] + var.dt/tau*sat_mix) / (1. + var.dt/tau)

        # evaporation: both tau < 0 and (y_NH3 - sat) < 0, so ice_loss > 0
        ice_loss = (var.y[:, species.index('NH3')] - sat_p)*var.dt/tau
        ice_loss = np.minimum(var.y[:, species.index('NH3_l_s')], ice_loss)

        var.ymix[conden_indx, species.index('NH3_l_s')] += (var.ymix[conden_indx, species.index('NH3')] - y_conden[conden_indx])
        var.ymix[conden_indx, species.index('NH3')]     = y_conden[conden_indx]

        var.ymix[evap_indx, species.index('NH3')]     += ice_loss[evap_indx]/atm.n_0[evap_indx]
        var.ymix[evap_indx, species.index('NH3_l_s')] -= ice_loss[evap_indx]/atm.n_0[evap_indx]

        var.ymix[:, species.index('NH3_l_s')] = np.maximum(var.ymix[:, species.index('NH3_l_s')], 0)

        var.y = var.ymix * np.vstack(np.sum(var.y[:, atm.gas_indx], axis=1))
        return var
