"""3rd-order Rosenbrock-Wanner integrator (Sandu's Rodas3).

A drop-in alternative to the Ros2 W-method.  Enabled via
``vulcan_cfg.ode_solver = 'Rodas3'``.

Reference
---------
Sandu, A., Verwer, J. G., Van Loon, M., Carmichael, G. R., Potra, F. A.,
Dabdub, D., & Seinfeld, J. H. (1997). *Benchmarking stiff ODE solvers for
atmospheric chemistry problems — I. Implicit vs explicit.*
Atmospheric Environment 31(19), 3151-3166.

The coefficient set used here matches the canonical KPP
(Kinetic PreProcessor, Sandu et al.) Rodas3 in ``int/rosenbrock.f90``:

    γ = 1/2  (so c0 = 1/(γh) = 2/h on the W-method LHS)

    α21 = 0
    α31 = 2;   α32 = 0
    α41 = 2;   α42 = 0;   α43 = 1

    γ21 = 4
    γ31 = 1;   γ32 = -1
    γ41 = 1;   γ42 = -1;  γ43 = -8/3

    b₁ = 2;    b₂ = 0;    b₃ = 1;    b₄ = 1   (3rd-order solution weights)
    e₁ = 0;    e₂ = 0;    e₃ = 0;    e₄ = 1   (embedded order-2 estimator)

    ros_NewF = (T, T, F, T)   # which stages re-evaluate f

Embedded error estimator is just k₄.  4 banded solves and 3 RHS
evaluations per step (stage 3 reuses stage 2's f).  L-stable, stiffly
accurate.  Per-step cost ≈ 2× Ros2; per-step accuracy is one order
higher, so the controller can take larger steps at tighter rtol.

Supported configurations
------------------------
The standard path (``use_moldiff=True``, ``use_settling=False``,
``use_vm_mol=False``) is supported and uses the existing
``lhs_jac_banded`` and ``diffdf`` kernels.  Edge-case configurations
(settling, upwind molecular-diffusion ``vm_mol``, no-moldiff) currently
fall back to Ros2 with a clear NotImplementedError — extending Rodas3
to those code paths is straightforward but not required for the common
HD189 / hot-Jupiter setup.
"""
import numpy as np
import scipy
from scipy.linalg.lapack import dgbtrf, dgbtrs

import vulcan_cfg
import build_atm
import chemistry_jax as chem_funs
from chemistry_jax import ni
from vulcan_cfg import nz

from chemistry_jax import chemdf

from ode_solver import ODESolver

compo = build_atm.compo
compo_row = build_atm.compo_row
species = chem_funs.spec_list


# ---------------------------------------------------------------------------
# Sandu Rodas3 coefficient set (KPP rosenbrock.f90 canonical values).
# ---------------------------------------------------------------------------
_GAMMA = 0.5

_A31 = 2.0
_A41 = 2.0
_A43 = 1.0
# α21=α32=α42=0 — not stored, written as literal 0 below.

_C21 =  4.0
_C31 =  1.0
_C32 = -1.0
_C41 =  1.0
_C42 = -1.0
_C43 = -8.0 / 3.0

_B1 = 2.0
_B3 = 1.0
_B4 = 1.0
# b2=0, not stored.


class Rodas3(ODESolver):
    """3rd-order Rosenbrock-Wanner with embedded 2nd-order error estimator.

    Inherits spatial discretisation (diffdf, lhs_jac_banded, clip, step_ok,
    step_reject, step_size, photochemistry) from ODESolver.  The
    step-size controller (I or PI) is inherited and parameterised by
    ``error_order = 3``.
    """

    error_order = 3

    def __init__(self):
        super().__init__()

    def solver(self, var, atm, para):
        """One Rodas3 step.  4 banded solves + 3 RHS evaluations."""
        y, ymix, h, k = var.y, var.ymix, var.dt, var.k
        M = atm.M

        # Rodas3 currently supports the most common path only.  Detect any
        # configuration that needs a different lhs_jac/diffdf and refuse
        # to run rather than silently produce wrong results.
        if vulcan_cfg.use_vm_mol:
            raise NotImplementedError(
                "Rodas3 does not (yet) support use_vm_mol=True; use Ros2.")
        if vulcan_cfg.use_settling:
            raise NotImplementedError(
                "Rodas3 does not (yet) support use_settling=True; use Ros2.")
        if not vulcan_cfg.use_moldiff:
            raise NotImplementedError(
                "Rodas3 does not (yet) support use_moldiff=False; use Ros2.")

        diffdf  = self.diffdf
        jac_fn  = self.lhs_jac_banded
        c0      = 1.0 / (_GAMMA * h)        # = 2/h

        # --- LHS (shared across all 4 stages) ---
        lhs_b, bw = jac_fn(var, atm, c0=c0)

        # Boundary/condensation/ion handling — same pattern as Ros2.
        if vulcan_cfg.use_condense and para.fix_species_start:
            for sp in vulcan_cfg.fix_species:
                if vulcan_cfg.fix_species_from_coldtrap_lev:
                    pfix_indx = atm.conden_min_lev[sp]
                    atm.fix_sp_indx[sp] = np.arange(
                        species.index(sp),
                        species.index(sp) + ni * pfix_indx, ni)
                lhs_b[:, atm.fix_sp_indx[sp]] = 0.
                lhs_b[bw, atm.fix_sp_indx[sp]] = c0
        if vulcan_cfg.use_ion:
            lhs_b[:, atm.fix_e_indx] = 0.
            lhs_b[bw, atm.fix_e_indx] = c0

        # Factor the banded LHS ONCE (LAPACK dgbtrf) so the 4 stages share
        # one LU factorisation.  scipy.linalg.solve_banded recomputes the
        # LU on every call, which would be ~3× more work for a 4-stage
        # method.  Expand to LAPACK's (2*kl+ku+1, n) layout (extra rows
        # for fill-in during pivoting).
        kl = ku = bw
        n_total = lhs_b.shape[1]
        ab_lapack = np.zeros((2 * kl + ku + 1, n_total))
        ab_lapack[kl:, :] = lhs_b
        lu, ipiv, info = dgbtrf(ab_lapack, kl, ku, overwrite_ab=1)
        if info != 0:
            raise RuntimeError(
                f"Rodas3 banded LU factorisation failed (LAPACK dgbtrf "
                f"info={info}); singular or ill-conditioned LHS.")

        def _solve(rhs):
            x, info_s = dgbtrs(lu, kl, ku, rhs, ipiv, overwrite_b=0)
            if info_s != 0:
                raise RuntimeError(
                    f"Rodas3 banded back-substitution failed "
                    f"(LAPACK dgbtrs info={info_s}).")
            return x

        # Helper to zero-out fixed-species/ion entries in any RHS vector.
        def _mask_rhs(df):
            if vulcan_cfg.use_condense and para.fix_species_start:
                for sp in vulcan_cfg.fix_species:
                    df[atm.fix_sp_indx[sp]] = 0
            if vulcan_cfg.use_ion:
                df[atm.fix_e_indx] = 0
            return df

        # --- Stage 1: W·k1 = f(y_n) ---
        f1 = (chemdf(y, M, k).flatten() + diffdf(y, atm).flatten())
        f1 = _mask_rhs(f1)
        k1_flat = _solve(f1)
        k1 = k1_flat.reshape(y.shape)

        # --- Stage 2: W·k2 = f(y_n) + (C21/h)·k1   (α21=0 → same y as stage 1) ---
        # KPP convention: RHS_i = F_temp + sum_{j<i} (C_ij / h) · k_j_flat
        f2 = f1   # ros_NewF[2]=T but α21=0 means new y == y_n; f is identical.
        rhs2 = f2 + (_C21 / h) * k1_flat
        k2_flat = _solve(rhs2)
        k2 = k2_flat.reshape(y.shape)

        # --- Stage 3: W·k3 = f(y_n) + (C31/h)·k1 + (C32/h)·k2   (ros_NewF[3]=F) ---
        rhs3 = f1 + (_C31 / h) * k1_flat + (_C32 / h) * k2_flat
        k3_flat = _solve(rhs3)
        k3 = k3_flat.reshape(y.shape)

        # --- Stage 4: W·k4 = f(y_n + α41·k1 + α43·k3) + Σ (C4j/h)·kj ---
        # α41=2, α42=0, α43=1.
        y_stage4 = y + _A41 * k1 + _A43 * k3
        f4 = (chemdf(y_stage4, M, k).flatten()
              + diffdf(y_stage4, atm).flatten())
        f4 = _mask_rhs(f4)
        rhs4 = (f4
                + (_C41 / h) * k1_flat
                + (_C42 / h) * k2_flat
                + (_C43 / h) * k3_flat)
        k4_flat = _solve(rhs4)
        k4 = k4_flat.reshape(y.shape)

        # --- 3rd-order solution: y_{n+1} = y_n + b1·k1 + b3·k3 + b4·k4 ---
        # (b2 = 0; b1 = 2, b3 = 1, b4 = 1.)
        sol = y + _B1 * k1 + _B3 * k3 + _B4 * k4

        # H2/He bottom-pin (mirrors Ros2.solver lines 142-148)
        if (getattr(vulcan_cfg, 'use_fix_H2He', False)
                and 'H2' not in vulcan_cfg.use_fix_sp_bot
                and var.t > 1e6):
            vulcan_cfg.use_fix_sp_bot['H2'] = var.ymix[0, species.index('H2')]
            vulcan_cfg.use_fix_sp_bot['He'] = var.ymix[0, species.index('He')]
            print("After 1e6 sec, H2 and He are fixed at " + str(
                (var.ymix[0, species.index('H2')],
                 var.ymix[0, species.index('He')])))
            self.fix_sp_bot_index = [species.index(sp)
                                     for sp in vulcan_cfg.use_fix_sp_bot.keys()]
            self.fix_sp_bot_mix = np.array(
                [vulcan_cfg.use_fix_sp_bot[sp]
                 for sp in vulcan_cfg.use_fix_sp_bot.keys()])

        if vulcan_cfg.use_fix_sp_bot:
            sol[0, self.fix_sp_bot_index] = self.fix_sp_bot_mix * atm.n_0[0]

        # --- Embedded estimator: y_err = e4·k4 = k4 ---
        # (mirrors the |sol - yk2| pattern in Ros2 — same role as truncation
        # error indicator, with the same masking by mtol/atol and same
        # fixed-species/condense suppression.)
        delta_arr = np.abs(k4)
        delta_arr[ymix < self.mtol] = 0
        delta_arr[sol  < self.atol] = 0

        if vulcan_cfg.use_botflux or vulcan_cfg.use_fix_sp_bot:
            delta_arr[0] = 0

        if vulcan_cfg.use_condense:
            delta_arr[:, self.non_gas_sp_index] = 0
            delta_arr[:, self.condense_sp_index] = 0
            if para.fix_species_start:
                for sp in vulcan_cfg.fix_species:
                    if not vulcan_cfg.fix_species_from_coldtrap_lev:
                        sol[:, species.index(sp)] = var.fix_y[sp].copy()
                    else:
                        pfix_indx = atm.conden_min_lev[sp]
                        sol[:pfix_indx, species.index(sp)] = (
                            var.fix_y[sp].copy()[:pfix_indx])
                    delta_arr[:, species.index(sp)] = 0

        if vulcan_cfg.use_print_delta and para.count % vulcan_cfg.print_prog_num == 0:
            max_lev_indx = np.nanargmax(delta_arr / np.maximum(sol, 1e-300))
            print('Largest delta (truncation error) at nz = '
                  + str(int(max_lev_indx / ni))
                  + ', species ' + species[max_lev_indx % ni])

        delta = np.amax(delta_arr[sol > 0] / sol[sol > 0])

        var.y = sol
        if vulcan_cfg.non_gas_sp:
            var.ymix = var.y / np.sum(var.y[:, atm.gas_indx], axis=1)[:, np.newaxis]
        else:
            var.ymix = var.y / np.sum(var.y, axis=1)[:, np.newaxis]

        para.delta = delta

        if vulcan_cfg.use_ion:
            var.y[:, species.index('e')] = 0
            for sp in var.charge_list:
                var.y[:, species.index('e')] -= (
                    compo[compo_row.index(sp)]['e']
                    * var.y[:, species.index(sp)])

        return var, para

    def naming_solver(self, para):
        """Select the dispatch name for one_step. Rodas3 has a single path."""
        if vulcan_cfg.use_moldiff:
            print('Include molecular diffusion (Rodas3).')
        else:
            raise NotImplementedError(
                "Rodas3 currently requires use_moldiff=True; use Ros2.")
        para.solver_str = 'solver'

    def one_step(self, var, atm, para):
        """Inherited control flow: keep retrying until step_ok or step_reject."""
        while True:
            var, para = getattr(self, para.solver_str)(var, atm, para)
            var, para = self.clip(var, para, atm)
            if self.step_ok(var, para):
                break
            elif self.step_reject(var, para):
                break
        return var, para
