"""JAX-side banded LHS Jacobian assembly for the Rosenbrock-2 solver.

The previous implementation in :class:`ODESolver.lhs_jac_banded` called the
JAX-jitted chemistry Jacobian (`chem_jac_blocks`), then performed ~70 lines
of Python+NumPy index arithmetic per step to scatter chemistry blocks into
banded form, add the c0 identity, and subtract eddy + molecular diffusion
terms (including bottom/top boundary conditions).

This module fuses all of that into a single JIT-compiled JAX kernel.  The
caller passes flat arrays; the kernel returns the banded matrix `ab` of
shape `(2*bw+1, nz*ni)` (with `bw = 2*ni-1`) ready for
`scipy.linalg.solve_banded`.

Numerical layout matches the NumPy implementation exactly so the regression
test gates correctness end-to-end.
"""

from functools import partial

import jax
import jax.numpy as jnp

from chemistry_jax import ni, _chemdf_single, _jac_vmap
from phy_const import kb, Navo


@partial(jax.jit, static_argnames=('nz',))
def _lhs_jac_banded_kernel(y, M, k, c0,
                           dzi, Kzz, Dzz, vz, vs, vm,
                           alpha, Tco, ms, g, Ti, Hpi,
                           gas_mask, bot_vdep,
                           use_botflux_flag, thermal_flag, vm_bot_flag,
                           nz):
    """Pure-JAX assembly of the banded LHS Jacobian = c0*I - dfdy.

    Pass c0 = 1/(r*dt) for the Rosenbrock LHS; pass c0 = 0 to obtain the
    pure steady-state Jacobian -dfdy used by the Newton finisher.

    Inputs
    ------
    y         : (nz, ni)        number density
    M         : (nz,)           total density
    k         : (nr+1, nz)      rate coefficients
    c0        : scalar          identity coefficient on the main diagonal
    dzi       : (nz-1,)         inter-layer spacings
    Kzz       : (nz-1,)         eddy diffusion
    Dzz       : (nz-1, ni)      molecular diffusion (per species);
                                pass zeros to disable mol-diff (no_mol path)
    vz        : (nz-1,)         vertical velocity at half-levels
    vs        : (nz-1, ni)      settling velocity at half-levels (per species);
                                pass zeros if settling disabled
    vm        : (nz, ni)        mean-molecular-velocity advection at cell centres
                                (per species); pass zeros if not used.  Caller is
                                responsible for zeroing vm[0] if vm should be
                                absent at the bottom (the settling_vm variant).
    alpha     : (ni,)           thermal diffusion factor
    Tco       : (nz,)           temperature at cell centres
    ms        : (ni,)           species molecular weight
    g         : (nz,)           gravity
    Ti        : (nz-1,)         interface temperature
    Hpi       : (nz-1,)         interface scale height
    gas_mask  : (ni,)           1.0 for gas species, 0.0 for non-gas
                                (use_condense=False  → all ones)
    bot_vdep  : (ni,)           deposition velocities at the bottom
    use_botflux_flag : 0.0 or 1.0
    thermal_flag     : 0.0 or 1.0   multiplies the mol-diff thermal drift bracket
                                    (-1/Hpi + ms*g/(Navo*kb*Ti) + alpha*dT/Ti).
                                    Pass 1.0 for the standard thermal mol-diff;
                                    pass 0.0 for the vm-advection variants where
                                    the drift is encoded in vm instead.
    vm_bot_flag      : 0.0 or 1.0   multiplies the vm contributions at the
                                    bottom boundary (j=0).  Pass 1.0 normally;
                                    pass 0.0 for the settling_vm variant where
                                    vm is absent at the bottom.  (Note: this
                                    zeroes only the j=0 boundary contribution;
                                    vm[0] is still used in the middle-layer
                                    j=1 upwind, matching the original numpy.)
    nz        : static int      number of vertical layers
    """
    bw = 2 * ni - 1

    # ------------------------------------------------------------------
    # 1. Chemistry Jacobian blocks (nz, ni, ni)
    #    banded position: ab[bw + si - sj, iz*ni + sj] = -jac[iz, si, sj]
    # ------------------------------------------------------------------
    jac = _jac_vmap(y, M, k)

    ab = jnp.zeros((2 * bw + 1, ni * nz))
    si, sj = jnp.mgrid[0:ni, 0:ni]                # (ni, ni)
    row_chem = bw + si - sj                       # (ni, ni)
    col_chem = jnp.arange(nz)[:, None, None] * ni + sj  # (nz, ni, ni)
    ab = ab.at[row_chem[None], col_chem].set(-jac)

    # ------------------------------------------------------------------
    # 2. Identity: add c0 to main diagonal (row bw)
    # ------------------------------------------------------------------
    ab = ab.at[bw].add(c0)

    # ------------------------------------------------------------------
    # 3. Diffusion: assemble contributions for the three active banded
    #    rows (bw, bw-ni, bw+ni) as (nz, ni) arrays, then add at the end.
    # ------------------------------------------------------------------
    ysum = (y * gas_mask).sum(axis=1)             # (nz,)

    # Upwind splits — avoid boolean masking, use clamps.
    vz_pos = jnp.maximum(vz, 0.0)                 # (vz>0)*vz       (nz-1,)
    vz_neg = jnp.minimum(vz, 0.0)                 # (vz<0)*vz       (nz-1,)
    vs_pos = jnp.maximum(vs, 0.0)                 # (nz-1, ni)
    vs_neg = jnp.minimum(vs, 0.0)                 # (nz-1, ni)
    vm_pos = jnp.maximum(vm, 0.0)                 # (nz, ni)
    vm_neg = jnp.minimum(vm, 0.0)                 # (nz, ni)

    diag_diff  = jnp.zeros((nz, ni))
    upper_diff = jnp.zeros((nz, ni))
    lower_diff = jnp.zeros((nz, ni))

    # ----- middle layers j = 1..nz-2 ---------------------------------
    j      = jnp.arange(1, nz - 1)                # (nz-2,)
    dz_ave = 0.5 * (dzi[j - 1] + dzi[j])          # (nz-2,)
    Dj     = Dzz[j]                                # (nz-2, ni)
    Dj1    = Dzz[j - 1]                            # (nz-2, ni)

    # eddy diffusion (scalar per layer)
    ek_d = (-1. / dz_ave * (Kzz[j] / dzi[j] * (ysum[j + 1] + ysum[j]) / 2.
                            + Kzz[j - 1] / dzi[j - 1] * (ysum[j - 1] + ysum[j]) / 2.)
            / ysum[j]
            - (vz_pos[j] - vz_neg[j - 1]) / dz_ave)
    ek_u = (1. / dz_ave * Kzz[j] / dzi[j] * (ysum[j + 1] + ysum[j])
            / (2. * ysum[j + 1])
            - vz_neg[j] / dz_ave)
    ek_l = (1. / dz_ave * Kzz[j - 1] / dzi[j - 1] * (ysum[j - 1] + ysum[j])
            / (2. * ysum[j - 1])
            + vz_pos[j - 1] / dz_ave)

    # molecular diffusion (per layer × per species)
    inv_dza  = 1. / dz_ave
    inv_dza2 = inv_dza / 2.
    dTj      = (Tco[j + 1] - Tco[j]) / dzi[j]
    dTj1     = (Tco[j]     - Tco[j - 1]) / dzi[j - 1]

    term_j = thermal_flag * Dj * (-1. / Hpi[j][:, None]
                   + ms * g[j][:, None] / (Navo * kb * Ti[j][:, None])
                   + alpha * dTj[:, None] / Ti[j][:, None])
    term_j1 = thermal_flag * Dj1 * (-1. / Hpi[j - 1][:, None]
                     + ms * g[j][:, None] / (Navo * kb * Ti[j - 1][:, None])
                     + alpha * dTj1[:, None] / Ti[j - 1][:, None])

    md_d_sc = (-inv_dza[:, None]
               * (Dj / dzi[j][:, None] * (ysum[j + 1] + ysum[j])[:, None] / 2.
                  + Dj1 / dzi[j - 1][:, None] * (ysum[j - 1] + ysum[j])[:, None] / 2.)
               / ysum[j][:, None])
    md_d = md_d_sc + inv_dza2[:, None] * (term_j - term_j1)

    term_u = thermal_flag * Dj * (-1. / Hpi[j][:, None]
                   + ms * g[j + 1][:, None] / (Navo * kb * Ti[j][:, None])
                   + alpha * dTj[:, None] / Ti[j][:, None])
    md_u = (inv_dza[:, None] * Dj / dzi[j][:, None]
            * (ysum[j + 1] + ysum[j])[:, None] / (2. * ysum[j + 1][:, None])
            + inv_dza2[:, None] * term_u)

    term_l = thermal_flag * Dj1 * (-1. / Hpi[j - 1][:, None]
                    + ms * g[j - 1][:, None] / (Navo * kb * Ti[j - 1][:, None])
                    + alpha * dTj1[:, None] / Ti[j - 1][:, None])
    md_l = (inv_dza[:, None] * Dj1 / dzi[j - 1][:, None]
            * (ysum[j - 1] + ysum[j])[:, None] / (2. * ysum[j - 1][:, None])
            - inv_dza2[:, None] * term_l)

    # vs (settling, interface-defined, per-species) upwind contributions.
    # Indexing matches vz since vs has shape (nz-1, ni).
    vs_d_mid = -(vs_pos[j] - vs_neg[j - 1]) / dz_ave[:, None]   # (nz-2, ni)
    vs_u_mid = -vs_neg[j] / dz_ave[:, None]
    vs_l_mid =  vs_pos[j - 1] / dz_ave[:, None]

    # vm (cell-centred mean-molecular-velocity advection, per-species) upwind
    # contributions; indexed at cell centres (vm has shape (nz, ni)).
    vm_d_mid = -(vm_pos[j] - vm_neg[j - 1]) / dz_ave[:, None]
    vm_u_mid = -vm_neg[j] / dz_ave[:, None]
    vm_l_mid =  vm_pos[j - 1] / dz_ave[:, None]

    md_d = md_d + vs_d_mid + vm_d_mid
    md_u = md_u + vs_u_mid + vm_u_mid
    md_l = md_l + vs_l_mid + vm_l_mid

    diag_diff  = diag_diff .at[1:nz - 1].add(-(ek_d[:, None] + md_d))
    upper_diff = upper_diff.at[2:nz    ].add(-(ek_u[:, None] + md_u))
    lower_diff = lower_diff.at[0:nz - 2].add(-(ek_l[:, None] + md_l))

    # ----- bottom BC (j = 0) -----------------------------------------
    mol_bc0 = thermal_flag * (-1. / Hpi[0] + ms * g[0] / (Navo * kb * Ti[0])
               + alpha / Ti[0] * (Tco[1] - Tco[0]) / dzi[0])               # (ni,)

    bot0_eddy = (-1. / dzi[0] * (Kzz[0] / dzi[0])
                 * (ysum[1] + ysum[0]) / (2. * ysum[0])
                 - vz_pos[0] / dzi[0])
    bot0_mol  = (-1. / dzi[0] * (Dzz[0] / dzi[0])
                 * (ysum[1] + ysum[0]) / (2. * ysum[0])
                 + 1. / dzi[0] * Dzz[0] / 2. * mol_bc0)                    # (ni,)
    # vs and vm bottom contributions (per-species)
    bot0_vs = -vs_pos[0] / dzi[0]                                          # (ni,)
    bot0_vm = vm_bot_flag * (-vm_pos[0] / dzi[0])                          # (ni,)
    diag_diff = diag_diff.at[0].add(-(bot0_eddy + bot0_mol + bot0_vs + bot0_vm))
    # use_botflux: ab_diag[0] -= -bot_vdep / dzi[0]
    diag_diff = diag_diff.at[0].add(use_botflux_flag * bot_vdep / dzi[0])

    bot1_eddy = (1. / dzi[0] * (Kzz[0] / dzi[0])
                 * (ysum[1] + ysum[0]) / (2. * ysum[1])
                 - vz_neg[0] / dzi[0])
    bot1_mol  = (1. / dzi[0] * (Dzz[0] / dzi[0])
                 * (ysum[1] + ysum[0]) / (2. * ysum[1])
                 + 1. / dzi[0] * Dzz[0] / 2. * mol_bc0)                    # (ni,)
    bot1_vs = -vs_neg[0] / dzi[0]                                          # (ni,)
    bot1_vm = vm_bot_flag * (-vm_neg[0] / dzi[0])                          # (ni,)
    upper_diff = upper_diff.at[1].add(-(bot1_eddy + bot1_mol + bot1_vs + bot1_vm))

    # ----- top BC (j = nz-1) -----------------------------------------
    mol_bcN = thermal_flag * (-1. / Hpi[-1] + ms * g[-1] / (Navo * kb * Ti[-1])
               + alpha / Ti[-1] * (Tco[-1] - Tco[-2]) / dzi[-1])           # (ni,)

    topN_eddy = (-1. / dzi[nz - 2] * (Kzz[nz - 2] / dzi[nz - 2])
                 * (ysum[nz - 2] + ysum[nz - 1]) / (2. * ysum[nz - 1])
                 + vz_neg[-1] / dzi[-1])
    topN_mol  = (-1. / dzi[nz - 2] * (Dzz[nz - 2] / dzi[nz - 2])
                 * (ysum[nz - 1] + ysum[nz - 2]) / (2. * ysum[nz - 1])
                 - 1. / dzi[-1] * Dzz[-1] / 2. * mol_bcN)                  # (ni,)
    # vs[-1] = vs[nz-2] (top interface, below cell nz-1).
    # vm[-1] = vm[nz-1] (top cell value).
    topN_vs = vs_neg[-1] / dzi[-1]                                         # (ni,)
    topN_vm = vm_neg[-1] / dzi[-1]                                         # (ni,)
    diag_diff = diag_diff.at[nz - 1].add(-(topN_eddy + topN_mol + topN_vs + topN_vm))

    topL_eddy = (1. / dzi[nz - 2] * (Kzz[nz - 2] / dzi[nz - 2])
                 * (ysum[nz - 2] + ysum[nz - 1]) / (2. * ysum[nz - 2])
                 + vz_pos[-1] / dzi[-1])
    topL_mol  = (1. / dzi[nz - 2] * (Dzz[nz - 2] / dzi[nz - 2])
                 * (ysum[nz - 1] + ysum[nz - 2]) / (2. * ysum[nz - 2])
                 - 1. / dzi[-1] * Dzz[-1] / 2. * mol_bcN)                  # (ni,)
    topL_vs = vs_pos[-1] / dzi[-1]                                         # (ni,)
    topL_vm = vm_pos[-1] / dzi[-1]                                         # (ni,)
    lower_diff = lower_diff.at[nz - 2].add(-(topL_eddy + topL_mol + topL_vs + topL_vm))

    # ------------------------------------------------------------------
    # 4. Add diffusion contributions to the three active banded rows
    # ------------------------------------------------------------------
    ab = ab.at[bw     ].add(diag_diff .reshape(-1))
    ab = ab.at[bw - ni].add(upper_diff.reshape(-1))
    ab = ab.at[bw + ni].add(lower_diff.reshape(-1))

    return ab
