import numpy as np

import vulcan_cfg
import chem_funs
from phy_const import kb, Navo
from vulcan_cfg import nz

species = chem_funs.spec_list


class Condensation:
    """Condensation and evaporation rate updates in the continuum diffusion regime."""

    def update(self, var, atm):
        """Update condensation reaction rate coefficients var.k for the current number densities."""
        for re in var.conden_re_list:
            if var.Rf[re] == 'H2O -> H2O_l_s' and 'H2O' in vulcan_cfg.condense_sp:
                if vulcan_cfg.use_relax:
                    var.k[re]   = np.repeat(0., nz)
                    var.k[re+1] = np.repeat(0., nz)
                else:
                    m     = 18./Navo
                    rho_p = atm.rho_p['H2O_l_s']
                    r_p   = atm.r_p['H2O_l_s']
                    sat_humidity = atm.sat_p['H2O']/kb/atm.Tco * vulcan_cfg.humidity

                    Dg   = np.insert(atm.Dzz[:, species.index('H2O')], 0, atm.Dzz[0, species.index('H2O')])
                    rate = Dg * m/rho_p /r_p**2 * (var.y[:, species.index('H2O')] - sat_humidity)

                    var.k[re]   = np.maximum(rate, 0)
                    var.k[re+1] = np.abs(np.minimum(rate, 0))

            elif var.Rf[re] == 'NH3 -> NH3_l' and 'NH3' in vulcan_cfg.condense_sp:
                if vulcan_cfg.use_relax:
                    var.k[re]   = np.repeat(0., nz)
                    var.k[re+1] = np.repeat(0., nz)
                else:
                    m     = 17./Navo
                    rho_p = atm.rho_p['NH3_l_s']
                    r_p   = atm.r_p['NH3_l_s']

                    Dg   = np.insert(atm.Dzz[:, species.index('NH3')], 0, atm.Dzz[0, species.index('NH3')])
                    rate = Dg * m/rho_p /r_p**2 * (var.y[:, species.index('NH3')] - atm.sat_p['NH3']/kb/atm.Tco)

                    var.k[re]   = np.maximum(rate, 0)
                    var.k[re+1] = np.abs(np.minimum(rate, 0))

            elif var.Rf[re] == 'H2SO4 -> H2SO4_l' and 'H2SO4' in vulcan_cfg.condense_sp:
                m     = 98.022/Navo
                rho_p = atm.rho_p['H2SO4_l']
                r_p   = atm.r_p['H2SO4_l']

                Dg   = np.insert(atm.Dzz[:, species.index('H2SO4')], 0, atm.Dzz[0, species.index('H2SO4')])
                rate = Dg * m/rho_p /r_p**2 * (var.y[:, species.index('H2SO4')] - atm.sat_p['H2SO4']/kb/atm.Tco)

                var.k[re]   = np.maximum(rate, 0)
                var.k[re+1] = np.abs(np.minimum(rate, 0))

            elif var.Rf[re] == 'S2 -> S2_l_s' and 'S2' in vulcan_cfg.condense_sp:
                m     = 45.019/Navo
                rho_p = atm.rho_p['S2_l_s']
                r_p   = atm.r_p['S2_l_s']

                Dg   = np.insert(atm.Dzz[:, species.index('S2')], 0, atm.Dzz[0, species.index('S2')])
                rate = Dg * m/rho_p /r_p**2 * (var.y[:, species.index('S2')] - atm.sat_p['S2']/kb/atm.Tco)

                var.k[re]   = np.maximum(rate, 0)
                var.k[re+1] = np.abs(np.minimum(rate, 0))

            elif var.Rf[re] == 'S4 -> S4_l_s' and 'S4' in vulcan_cfg.condense_sp:
                m     = 32.06*4/Navo
                rho_p = atm.rho_p['S4_l_s']
                r_p   = atm.r_p['S4_l_s']

                Dg   = np.insert(atm.Dzz[:, species.index('S4')], 0, atm.Dzz[0, species.index('S4')])
                rate = Dg * m/rho_p /r_p**2 * (var.y[:, species.index('S4')] - atm.sat_p['S4']/kb/atm.Tco)

                var.k[re]   = np.maximum(rate, 0)
                var.k[re+1] = np.abs(np.minimum(rate, 0))

            elif var.Rf[re] == 'S8 -> S8_l_s' and 'S8' in vulcan_cfg.condense_sp:
                m     = 360.152/Navo
                rho_p = atm.rho_p['S8_l_s']
                r_p   = atm.r_p['S8_l_s']

                Dg   = np.insert(atm.Dzz[:, species.index('S8')], 0, atm.Dzz[0, species.index('S8')])
                rate = Dg * m/rho_p /r_p**2 * (var.y[:, species.index('S8')] - atm.sat_p['S8']/kb/atm.Tco)

                var.k[re]   = np.maximum(rate, 0)
                var.k[re+1] = np.abs(np.minimum(rate, 0))

            elif var.Rf[re] == 'C -> C_s' and 'C' in vulcan_cfg.condense_sp:
                m     = 12.011/Navo
                rho_p = atm.rho_p['C_s']
                r_p   = atm.r_p['C_s']

                Dg   = np.insert(atm.Dzz[:, species.index('C')], 0, atm.Dzz[0, species.index('C')])
                rate = Dg * m/rho_p /r_p**2 * (var.y[:, species.index('C')] - atm.sat_p['C']/kb/atm.Tco)

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
