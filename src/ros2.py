import numpy as np
from scipy.linalg.lapack import dgbtrf, dgbtrs

from neovulcan_runtime import get_cfg
cfg = get_cfg()
import build_atm
import chemistry_jax as chem_funs
from chemistry_jax import ni
nz = cfg.atmosphere.nz

from chemistry_jax import chemdf

from ode_solver import ODESolver, zero_rows_banded

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

        # diffdf still uses the numpy variants (only ~1.5 ms/step total, so
        # not worth porting yet); the LHS Jacobian goes through the unified
        # fused JAX kernel `lhs_jac_banded`, which routes settling/vm/no_mol
        # via the cached vs/vm/Dzz/thermal_flag/vm_bot_flag arrays.
        if not cfg.atmosphere.use_vm_mol:
            if cfg.atmosphere.use_moldiff and not cfg.condensation.use_settling:
                diffdf = self.diffdf
            elif cfg.atmosphere.use_moldiff and cfg.condensation.use_settling:
                diffdf = self.diffdf_settling
            else:
                diffdf = self.diffdf_no_mol
        else:
            if cfg.atmosphere.use_moldiff and not cfg.condensation.use_settling:
                diffdf = self.diffdf_vm
            elif cfg.atmosphere.use_moldiff and cfg.condensation.use_settling:
                diffdf = self.diffdf_settling_vm
            else:
                diffdf = self.diffdf_no_mol

        jac_fn     = self.lhs_jac_banded
        use_banded = True

        r = 1. + 1./2.**0.5

        df = chemdf(y, M, k).flatten() + diffdf(y, atm).flatten()

        if use_banded:
            lhs_b, bw = jac_fn(var, atm)
            # lhs_b is in LAPACK band storage (3*bw+1 rows; main diagonal at row 2*bw).
            if cfg.condensation.use_condense and para.fix_species_start:
                for sp in cfg.condensation.fix_species:
                    if not cfg.condensation.fix_species_from_coldtrap_lev:
                        pass
                    else:
                        pfix_indx = atm.conden_min_lev[sp]
                        atm.fix_sp_indx[sp] = np.arange(species.index(sp), species.index(sp) + ni*(pfix_indx), ni)
                    df[atm.fix_sp_indx[sp]] = 0
                    zero_rows_banded(lhs_b, bw, atm.fix_sp_indx[sp], 1./(r*h))
            if cfg.photochemistry.use_ion:
                df[atm.fix_e_indx] = 0
                zero_rows_banded(lhs_b, bw, atm.fix_e_indx, 1./(r*h))
        else:
            lhs = jac_fn(var, atm)
            if cfg.condensation.use_condense and para.fix_species_start:
                for sp in cfg.condensation.fix_species:
                    if not cfg.condensation.fix_species_from_coldtrap_lev:
                        pass
                    else:
                        pfix_indx = atm.conden_min_lev[sp]
                        atm.fix_sp_indx[sp] = np.arange(species.index(sp), species.index(sp) + ni*(pfix_indx), ni)
                    df[atm.fix_sp_indx[sp]] = 0
                    lhs[atm.fix_sp_indx[sp], :] = 0
                    lhs[atm.fix_sp_indx[sp], atm.fix_sp_indx[sp]] = 1./(r*h)
            if cfg.photochemistry.use_ion:
                df[atm.fix_e_indx] = 0
                lhs[atm.fix_e_indx, :] = 0
                lhs[atm.fix_e_indx, atm.fix_e_indx] = 1./(r*h)
            lhs_b, bw = self.store_bandM(lhs, ni, nz)

        # Factor lhs_b once (dgbtrf, in-place), then back-substitute twice
        # (one dgbtrs per stage — the W-method shares the same LHS).  Old code
        # called scipy.linalg.solve_banded twice, which re-factored on each
        # call; factor cost dominates the solve at bw=2*ni-1, so reuse here
        # is a ~25-30% per-step win.  lhs_b is already in LAPACK band storage
        # (lhs_jac_banded materialised it directly), so no padding needed.
        ab_factored, ipiv, info = dgbtrf(lhs_b, bw, bw, overwrite_ab=1)
        if info != 0:
            raise RuntimeError(f"dgbtrf failed: info={info}")
        k1_flat, info = dgbtrs(ab_factored, bw, bw, df, ipiv)
        if info != 0:
            raise RuntimeError(f"dgbtrs (k1) failed: info={info}")
        k1 = k1_flat.reshape(y.shape)

        yk2 = y + k1/r
        df = chemdf(yk2, M, k).flatten() + diffdf(yk2, atm).flatten()

        if cfg.condensation.use_condense and para.fix_species_start:
            for sp in cfg.condensation.fix_species:
                df[atm.fix_sp_indx[sp]] = 0
        if cfg.photochemistry.use_ion:
            df[atm.fix_e_indx] = 0

        rhs = df - 2./(r*h)*k1_flat
        k2, info = dgbtrs(ab_factored, bw, bw, rhs, ipiv)
        if info != 0:
            raise RuntimeError(f"dgbtrs (k2) failed: info={info}")
        k2 = k2.reshape(y.shape)

        sol = y + 3./(2.*r)*k1 + 1/(2.*r)*k2

        if cfg.boundary_conditions.use_fix_H2He and 'H2' not in cfg.boundary_conditions.use_fix_sp_bot and var.t > 1e6:
            cfg.boundary_conditions.use_fix_sp_bot['H2'] = var.ymix[0, species.index('H2')]
            cfg.boundary_conditions.use_fix_sp_bot['He'] = var.ymix[0, species.index('He')]
            print("After 1e6 sec, H2 and He are fixed at "
                  + str((var.ymix[0, species.index('H2')], var.ymix[0, species.index('He')])))
            self.fix_sp_bot_index = [species.index(sp) for sp in cfg.boundary_conditions.use_fix_sp_bot.keys()]
            self.fix_sp_bot_mix = np.array([cfg.boundary_conditions.use_fix_sp_bot[sp] for sp in cfg.boundary_conditions.use_fix_sp_bot.keys()])

        if cfg.boundary_conditions.use_fix_sp_bot:
            sol[0, self.fix_sp_bot_index] = self.fix_sp_bot_mix * atm.n_0[0]

        delta = np.abs(sol - yk2)
        delta[ymix < self.mtol] = 0
        delta[sol < self.atol] = 0

        if cfg.boundary_conditions.use_botflux or cfg.boundary_conditions.use_fix_sp_bot:
            delta[0] = 0

        if cfg.condensation.use_condense:
            delta[:, self.non_gas_sp_index] = 0
            delta[:, self.condense_sp_index] = 0

            if para.fix_species_start:
                for sp in cfg.condensation.fix_species:
                    if not cfg.condensation.fix_species_from_coldtrap_lev:
                        sol[:, species.index(sp)] = var.fix_y[sp].copy()
                    else:
                        pfix_indx = atm.conden_min_lev[sp]
                        sol[:pfix_indx, species.index(sp)] = var.fix_y[sp].copy()[:pfix_indx]
                    delta[:, species.index(sp)] = 0

        if cfg.solver.use_print_delta and para.count % cfg.solver.print_prog_num == 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(sol > 0, delta / sol, 0.)
            top = np.argsort(-ratio, axis=None)[:5]
            print('Largest delta (truncation error) contributors (species, nz, delta/y):')
            for flat in top:
                lev, isp = divmod(int(flat), ni)
                print(f'    {species[isp]:10s} nz = {lev:3d}   {ratio.flat[flat]:.2e}   y = {sol.flat[flat]:.2e}')

        delta = np.amax(delta[sol > 0] / sol[sol > 0])

        var.y = sol

        if cfg.condensation.non_gas_sp:
            var.ymix = var.y / np.sum(var.y[:, atm.gas_indx], axis=1)[:, np.newaxis]
        else:
            var.ymix = var.y / np.sum(var.y, axis=1)[:, np.newaxis]

        para.delta = delta

        if cfg.photochemistry.use_ion:
            var.y[:, species.index('e')] = 0
            for sp in var.charge_list:
                var.y[:, species.index('e')] -= compo[compo_row.index(sp)]['e'] * var.y[:, species.index(sp)]

        return var, para

    def solver_fix_all_bot(self, var, atm, para):
        """2nd-order Rosenbrock step with fixed bottom boundary condition."""
        y, ymix, h, k = var.y, var.ymix, var.dt, var.k
        M, dzi, Kzz = atm.M, atm.dzi, atm.Kzz

        bottom = np.copy(ymix[0])

        if cfg.atmosphere.use_moldiff:
            diffdf  = self.diffdf
            jac_tot = self.lhs_jac_fix_all_bot
        else:
            diffdf  = self.diffdf_no_mol
            jac_tot = self.lhs_jac_no_mol_fix_all_bot

        r = 1. + 1./2.**0.5

        df = chemdf(y, M, k).flatten() + diffdf(y, atm).flatten()
        lhs = jac_tot(var, atm)

        lhs_b, bw = self.store_bandM(lhs, ni, nz)

        # Factor once, back-substitute twice — same trick as in solver().
        N = lhs_b.shape[1]
        ab_lapack = np.empty((3 * bw + 1, N))
        ab_lapack[:bw] = 0.0
        ab_lapack[bw:] = lhs_b
        ab_factored, ipiv, info = dgbtrf(ab_lapack, bw, bw, overwrite_ab=1)
        if info != 0:
            raise RuntimeError(f"dgbtrf failed: info={info}")
        k1_flat, info = dgbtrs(ab_factored, bw, bw, df, ipiv)
        if info != 0:
            raise RuntimeError(f"dgbtrs (k1) failed: info={info}")
        k1 = k1_flat.reshape(y.shape)

        yk2 = y + k1/r
        df = chemdf(yk2, M, k).flatten() + diffdf(yk2, atm).flatten()

        rhs = df - 2./(r*h)*k1_flat
        k2, info = dgbtrs(ab_factored, bw, bw, rhs, ipiv)
        if info != 0:
            raise RuntimeError(f"dgbtrs (k2) failed: info={info}")
        k2 = k2.reshape(y.shape)

        sol = y + 3./(2.*r)*k1 + 1/(2.*r)*k2

        sol[0] = bottom * atm.n_0[0]

        delta = np.abs(sol - yk2)
        delta[ymix < self.mtol] = 0
        delta[sol < self.atol] = 0

        delta = np.amax(delta[sol > 0] / sol[sol > 0])

        var.y = sol

        if cfg.condensation.non_gas_sp:
            var.ymix = var.y / np.sum(var.y[:, atm.gas_indx], axis=1)[:, np.newaxis]
        else:
            var.ymix = var.y / np.sum(var.y, axis=1)[:, np.newaxis]

        para.delta = delta

        if cfg.photochemistry.use_ion:
            var.y[:, species.index('e')] = 0
            for sp in var.charge_list:
                var.y[:, species.index('e')] -= compo[compo_row.index(sp)]['e'] * var.y[:, species.index(sp)]

        return var, para

    def naming_solver(self, para):
        if cfg.atmosphere.use_moldiff:
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

    # step_size is provided by the ODESolver base class (handles both the
    # legacy I-controller and the optional Gustafsson PI controller via
    # cfg.solver.use_pi_controller).  Ros2 inherits error_order = 2.
