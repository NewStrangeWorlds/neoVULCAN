import numpy as np
import scipy

import vulcan_cfg
import build_atm
import chem_funs
from chem_funs import ni
from vulcan_cfg import nz

from chemistry_jax import chemdf

from ode_solver import ODESolver

compo = build_atm.compo
compo_row = build_atm.compo_row
species = chem_funs.spec_list


class Ros2(ODESolver):
    """2nd-order Rosenbrock time integrator [Verwer et al. 1997].

    Inherits spatial discretisation (diffdf, lhs_jac variants, clip, step_ok,
    step_reject, photochemistry) from ODESolver and adds the time-stepping logic.
    """

    def __init__(self):
        super().__init__()

    def store_bandM(self, a, nb, nn):
        """Convert a dense block-tridiagonal matrix to scipy banded form.

        a  : square block-tridiagonal matrix
        nb : block size (number of species)
        nn : number of blocks (number of layers)
        Returns (ab, bw) ready for scipy.linalg.solve_banded.
        """
        bw = 2*nb - 1
        ab = np.zeros((2*bw + 1, nb*nn))

        for i in range(0, 2*nb):
            ab[-(2*nb+i):, i] = a[0:2*nb+i, i]

        for i in range(2*nb, nn*nb - 2*nb):
            ab[:, i] = a[(i-2*nb+1):(i-2*nb+1)+(2*bw+1), i]

        for ne, i in enumerate(range(nn*nb - 2*nb, nn*nb)):
            ab[:(2*bw+1-ne), i] = a[-(2*bw+1-ne):, i]

        return ab, bw

    def solver(self, var, atm, para):
        """2nd-order Rosenbrock step with banded-matrix solve.

        Dispatches to the correct diffdf/lhs_jac pair based on vulcan_cfg flags.
        """
        y, ymix, h, k = var.y, var.ymix, var.dt, var.k
        M, dzi, Kzz = atm.M, atm.dzi, atm.Kzz

        if vulcan_cfg.use_vm_mol == False:
            if vulcan_cfg.use_moldiff == True and vulcan_cfg.use_settling == False:
                diffdf     = self.diffdf
                jac_fn     = self.lhs_jac_banded
                use_banded = True
            elif vulcan_cfg.use_moldiff == True and vulcan_cfg.use_settling == True:
                diffdf     = self.diffdf_settling
                jac_fn     = self.lhs_jac_settling
                use_banded = False
            else:
                diffdf     = self.diffdf_no_mol
                jac_fn     = self.lhs_jac_no_mol
                use_banded = False
        else:
            if vulcan_cfg.use_moldiff == True and vulcan_cfg.use_settling == False:
                diffdf     = self.diffdf_vm
                jac_fn     = self.lhs_jac_tot_vm
                use_banded = False
            elif vulcan_cfg.use_moldiff == True and vulcan_cfg.use_settling == True:
                diffdf     = self.diffdf_settling_vm
                jac_fn     = self.lhs_jac_settling_vm
                use_banded = False
            else:
                diffdf     = self.diffdf_no_mol
                jac_fn     = self.lhs_jac_no_mol
                use_banded = False

        r = 1. + 1./2.**0.5

        df = chemdf(y, M, k).flatten() + diffdf(y, atm).flatten()

        if use_banded:
            lhs_b, bw = jac_fn(var, atm)
            if vulcan_cfg.use_condense == True and para.fix_species_start == True:
                for sp in vulcan_cfg.fix_species:
                    if vulcan_cfg.fix_species_from_coldtrap_lev == False:
                        pass
                    else:
                        pfix_indx = atm.conden_min_lev[sp]
                        atm.fix_sp_indx[sp] = np.arange(species.index(sp), species.index(sp) + ni*(pfix_indx), ni)
                    df[atm.fix_sp_indx[sp]] = 0
                    lhs_b[:, atm.fix_sp_indx[sp]] = 0.
                    lhs_b[bw, atm.fix_sp_indx[sp]] = 1./(r*h)
            if vulcan_cfg.use_ion == True:
                df[atm.fix_e_indx] = 0
                lhs_b[:, atm.fix_e_indx] = 0.
                lhs_b[bw, atm.fix_e_indx] = 1./(r*h)
        else:
            lhs = jac_fn(var, atm)
            if vulcan_cfg.use_condense == True and para.fix_species_start == True:
                for sp in vulcan_cfg.fix_species:
                    if vulcan_cfg.fix_species_from_coldtrap_lev == False:
                        pass
                    else:
                        pfix_indx = atm.conden_min_lev[sp]
                        atm.fix_sp_indx[sp] = np.arange(species.index(sp), species.index(sp) + ni*(pfix_indx), ni)
                    df[atm.fix_sp_indx[sp]] = 0
                    lhs[atm.fix_sp_indx[sp], :] = 0
                    lhs[atm.fix_sp_indx[sp], atm.fix_sp_indx[sp]] = 1./(r*h)
            if vulcan_cfg.use_ion == True:
                df[atm.fix_e_indx] = 0
                lhs[atm.fix_e_indx, :] = 0
                lhs[atm.fix_e_indx, atm.fix_e_indx] = 1./(r*h)
            lhs_b, bw = self.store_bandM(lhs, ni, nz)

        k1_flat = scipy.linalg.solve_banded((bw, bw), lhs_b, df)
        k1 = k1_flat.reshape(y.shape)

        yk2 = y + k1/r
        df = chemdf(yk2, M, k).flatten() + diffdf(yk2, atm).flatten()

        if vulcan_cfg.use_condense == True and para.fix_species_start == True:
            for sp in vulcan_cfg.fix_species:
                df[atm.fix_sp_indx[sp]] = 0
        if vulcan_cfg.use_ion == True:
            df[atm.fix_e_indx] = 0

        rhs = df - 2./(r*h)*k1_flat
        k2 = scipy.linalg.solve_banded((bw, bw), lhs_b, rhs)
        k2 = k2.reshape(y.shape)

        sol = y + 3./(2.*r)*k1 + 1/(2.*r)*k2

        if getattr(vulcan_cfg, 'use_fix_H2He', False) and 'H2' not in vulcan_cfg.use_fix_sp_bot and var.t > 1e6:
            vulcan_cfg.use_fix_sp_bot['H2'] = var.ymix[0, species.index('H2')]
            vulcan_cfg.use_fix_sp_bot['He'] = var.ymix[0, species.index('He')]
            print("After 1e6 sec, H2 and He are fixed at "
                  + str((var.ymix[0, species.index('H2')], var.ymix[0, species.index('He')])))
            self.fix_sp_bot_index = [species.index(sp) for sp in vulcan_cfg.use_fix_sp_bot.keys()]
            self.fix_sp_bot_mix = np.array([vulcan_cfg.use_fix_sp_bot[sp] for sp in vulcan_cfg.use_fix_sp_bot.keys()])

        if vulcan_cfg.use_fix_sp_bot:
            sol[0, self.fix_sp_bot_index] = self.fix_sp_bot_mix * atm.n_0[0]

        delta = np.abs(sol - yk2)
        delta[ymix < self.mtol] = 0
        delta[sol < self.atol] = 0

        if vulcan_cfg.use_botflux == True or vulcan_cfg.use_fix_sp_bot:
            delta[0] = 0

        if vulcan_cfg.use_condense == True:
            delta[:, self.non_gas_sp_index] = 0
            delta[:, self.condense_sp_index] = 0

            if para.fix_species_start == True:
                for sp in vulcan_cfg.fix_species:
                    if vulcan_cfg.fix_species_from_coldtrap_lev == False:
                        sol[:, species.index(sp)] = var.fix_y[sp].copy()
                    else:
                        pfix_indx = atm.conden_min_lev[sp]
                        sol[:pfix_indx, species.index(sp)] = var.fix_y[sp].copy()[:pfix_indx]
                    delta[:, species.index(sp)] = 0

        if vulcan_cfg.use_print_delta == True and para.count % vulcan_cfg.print_prog_num == 0:
            max_indx = np.nanargmax(delta/sol, axis=1)
            max_lev_indx = np.nanargmax(delta/sol)
            print('Largest delta (truncation error) from nz = ' + str(int(max_lev_indx/ni)))
            print(np.array(species)[max_indx])
            print('Largest delta (truncation error) from ' + species[max_indx%ni]
                  + " at nz = " + str(int(max_indx/ni)))

        delta = np.amax(delta[sol > 0] / sol[sol > 0])

        var.y = sol

        if vulcan_cfg.non_gas_sp:
            var.ymix = var.y / np.vstack(np.sum(var.y[:, atm.gas_indx], axis=1))
        else:
            var.ymix = var.y / np.vstack(np.sum(var.y, axis=1))

        para.delta = delta

        if vulcan_cfg.use_ion == True:
            var.y[:, species.index('e')] = 0
            for sp in var.charge_list:
                var.y[:, species.index('e')] -= compo[compo_row.index(sp)]['e'] * var.y[:, species.index(sp)]

        return var, para

    def solver_fix_all_bot(self, var, atm, para):
        """2nd-order Rosenbrock step with fixed bottom boundary condition."""
        y, ymix, h, k = var.y, var.ymix, var.dt, var.k
        M, dzi, Kzz = atm.M, atm.dzi, atm.Kzz

        bottom = np.copy(ymix[0])

        if vulcan_cfg.use_moldiff == True:
            diffdf  = self.diffdf
            jac_tot = self.lhs_jac_fix_all_bot
        else:
            diffdf  = self.diffdf_no_mol
            jac_tot = self.lhs_jac_no_mol_fix_all_bot

        r = 1. + 1./2.**0.5

        df = chemdf(y, M, k).flatten() + diffdf(y, atm).flatten()
        lhs = jac_tot(var, atm)

        lhs_b, bw = self.store_bandM(lhs, ni, nz)
        k1_flat = scipy.linalg.solve_banded((bw, bw), lhs_b, df)
        k1 = k1_flat.reshape(y.shape)

        yk2 = y + k1/r
        df = chemdf(yk2, M, k).flatten() + diffdf(yk2, atm).flatten()

        rhs = df - 2./(r*h)*k1_flat
        k2 = scipy.linalg.solve_banded((bw, bw), lhs_b, rhs)
        k2 = k2.reshape(y.shape)

        sol = y + 3./(2.*r)*k1 + 1/(2.*r)*k2

        sol[0] = bottom * atm.n_0[0]

        delta = np.abs(sol - yk2)
        delta[ymix < self.mtol] = 0
        delta[sol < self.atol] = 0

        delta = np.amax(delta[sol > 0] / sol[sol > 0])

        var.y = sol

        if vulcan_cfg.non_gas_sp:
            var.ymix = var.y / np.vstack(np.sum(var.y[:, atm.gas_indx], axis=1))
        else:
            var.ymix = var.y / np.vstack(np.sum(var.y, axis=1))

        para.delta = delta

        if vulcan_cfg.use_ion == True:
            var.y[:, species.index('e')] = 0
            for sp in var.charge_list:
                var.y[:, species.index('e')] -= compo[compo_row.index(sp)]['e'] * var.y[:, species.index(sp)]

        return var, para

    def naming_solver(self, para):
        if vulcan_cfg.use_moldiff == True:
            print('Include molecular diffusion.')
        else:
            print('No molecular diffusion.')
        para.solver_str = 'solver'

    def one_step(self, var, atm, para):
        while True:
            var, para = getattr(self, para.solver_str)(var, atm, para)
            var, para = self.clip(var, para, atm)
            if self.step_ok(var, para):
                break
            elif self.step_reject(var, para):
                break
        return var, para

    def step_size(self, var, para,
                  dt_var_min=vulcan_cfg.dt_var_min, dt_var_max=vulcan_cfg.dt_var_max,
                  dt_min=vulcan_cfg.dt_min, dt_max=vulcan_cfg.dt_max):
        """Step-size control by truncation error estimate."""
        h = var.dt
        delta = para.delta
        rtol = vulcan_cfg.rtol

        if delta == 0:
            delta = 0.01 * rtol
        h_factor = 0.9 * (rtol/delta)**0.5
        h_factor = np.maximum(h_factor, dt_var_min)
        h_factor = np.minimum(h_factor, dt_var_max)

        h *= h_factor
        h = np.maximum(h, dt_min)
        h = np.minimum(h, dt_max)

        var.dt = h
        return var
