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


# ===========================================================================
# Log-space chemistry RHS and Jacobian
#
# DORMANT INFRASTRUCTURE: validated to machine precision against the manual
# chain-rule transform of the linear-y chemistry Jacobian.  Currently NOT
# wired into the integration loop — the log-space integrator (Tier 2.2 Phase
# C) was reverted after discovering that the chain rule cancels the natural
# implicit-damping of first-order chemistry loss (see plan / commit log).
# Kept here as ready-to-use kernels for the next iteration of the stiffness
# work (likely exponential Rosenbrock, where the linear stiff part is
# treated exactly and these kernels may still be useful for the remaining
# nonlinear part if a log-space variant is chosen).
#
# Math reference (chain rule):
#     g(x) = chemdf(exp(x), M, k) / exp(x)
# JAX autodiff (jax.jacfwd) applies the chain rule automatically — no
# manual J_x = J_y * y_j/y_i transform in user code.  The resulting
# log-space Jacobian has the same sparsity as the linear-y one.
# ===========================================================================

# x is clamped on both sides to keep exp(x) inside float64 range while still
# allowing the integrator to take large transient log-steps that would
# otherwise be rejected.  -300 → exp ≈ 5e-131 (well below any physical
# concentration), +200 → exp ≈ 7e86 (well above any physical concentration
# in giant-planet deep atmospheres).  See plan section "Edge cases".
_X_FLOOR = -300.0
_X_CEIL  =  200.0


def _chemdf_logy_single(x, M, k):
    """Per-cell log-space chemistry RHS: g(x) = chemdf(exp(x)) / exp(x).

    ``x`` is clipped to ``[_X_FLOOR, _X_CEIL]`` to guarantee finite exp(x).
    Inside the autodiff this means species pinned at the clamp see zero
    Jacobian sensitivity — which correctly reflects "this iterate is in
    the unphysical regime and the integrator's step controller should
    reject it."
    """
    x_safe = jnp.clip(x, _X_FLOOR, _X_CEIL)
    y = jnp.exp(x_safe)
    return _chemdf_single(y, M, k) / y


_chemdf_logy_vmap = jax.vmap(_chemdf_logy_single, in_axes=(0, 0, 1))
chemdf_logy_jax   = jax.jit(_chemdf_logy_vmap)

_jac_logy_single  = jax.jacfwd(_chemdf_logy_single, argnums=0)
_jac_logy_vmap    = jax.vmap(_jac_logy_single, in_axes=(0, 0, 1))
_jac_logy_jit     = jax.jit(_jac_logy_vmap)


def chemdf_logy(x, M, k_dict):
    """NumPy-facing wrapper for the log-space chemistry RHS.

    Mirrors :func:`chemistry_jax.chemdf` but accepts the log-space state
    ``x = log(y)`` and returns ``chemdf(exp(x), M, k) / exp(x)`` as a
    NumPy array.
    """
    import numpy as _np
    from chemistry_jax import k_dict_to_array
    k_arr = k_dict_to_array(k_dict)
    return _np.asarray(chemdf_logy_jax(jnp.asarray(x),
                                       jnp.asarray(M),
                                       jnp.asarray(k_arr)))


# ===========================================================================
# Per-species first-order loss rates k_loss_i^j and the phi_1 function.
#
# DORMANT INFRASTRUCTURE intended for a future exponential Rosenbrock
# integrator.  An exp-Euler prototype was attempted this session and shown
# *not* to work for VULCAN because diffusion needs implicit treatment
# (CFL-unstable when handled explicitly at the dt sizes the chemistry would
# allow).  See plan file for the full IMEX exp-Rosenbrock scope.
#
# What stays here as ready-to-use building blocks:
#   * chem_loss_rates(y, M, k_dict) → (nz, ni)
#       The diagonal first-order loss rate per (layer, species).  Non-negative
#       everywhere.  Computed from -diag(J_y_chem); validated to give the
#       expected stiff value (1.7e+17 for P4O6 dimerization at deep layer 0).
#   * phi_1(z) = (e^z - 1) / z, with phi_1(0) = 1
#       Numerically stable; validated to machine precision from |z| ~ 1e-20
#       up to z = -1e6.  Building block for any exponential integrator.
#
# Math reference for chem_loss_rates: for elementary chemistry, the diagonal
# J_y[ii] = ∂(dn_i/dt) / ∂y_i contains exactly the contributions of
# first-order loss reactions plus (rare) autocatalytic self-production.  In
# neutral-species photochemistry there is no autocatalysis so the diagonal
# is non-positive and `k_loss_i = -diag(J_y_chem)[i]`.
# ===========================================================================

@jax.jit
def _chem_loss_diag(y, M, k):
    """Diagonal of the chemistry Jacobian, per (layer, species)."""
    jac = _jac_vmap(y, M, k)                  # (nz, ni, ni)
    return jnp.einsum('ijj->ij', jac)         # (nz, ni)


def chem_loss_rates(y, M, k_dict):
    """Return per-cell first-order loss rates k_loss_i^j as NumPy (nz, ni).

    Computed from -diag(J_y_chem).  Clamped at 0 to guard against the rare
    case of an autocatalytic production term that would make a diagonal
    entry positive (no such reactions in VULCAN's NCHO/P-network, but the
    clamp is cheap insurance).

    Used by the exponential Rosenbrock integrator's diagonal phi_1
    treatment of stiff first-order loss.
    """
    import numpy as _np
    from chemistry_jax import k_dict_to_array
    k_arr = k_dict_to_array(k_dict)
    diag = _np.asarray(_chem_loss_diag(jnp.asarray(y),
                                       jnp.asarray(M),
                                       jnp.asarray(k_arr)))
    return _np.maximum(-diag, 0.0)


def phi_1(z):
    """phi_1(z) = (e^z - 1) / z, with phi_1(0) = 1.

    Numerically stable for all z: uses np.expm1 to avoid catastrophic
    cancellation in the numerator, and a Taylor-series branch for very
    small |z| where the direct formula's relative error still becomes
    visible.  In the exp-Rosenbrock context we evaluate at z = -k_loss·dt
    which is non-positive, but the implementation is general.
    """
    import numpy as _np
    z = _np.asarray(z)
    # For |z| < 1e-4, the direct formula loses ~4 digits of precision;
    # the Taylor series 1 + z/2 + z²/6 + z³/24 is accurate to ~16 digits.
    small = _np.abs(z) < 1e-4
    z_safe = _np.where(small, 1.0, z)  # avoid 0/0 inside the direct branch
    direct = _np.expm1(z) / z_safe
    taylor = 1.0 + z * (0.5 + z * (1.0/6.0 + z * (1.0/24.0)))
    return _np.where(small, taylor, direct)


def phi_2(z):
    """phi_2(z) = (e^z - 1 - z) / z², with phi_2(0) = 1/2.

    The direct formula (expm1(z) - z) / z² suffers catastrophic
    cancellation for |z| up to ~1e-2: expm1(z) - z ≈ z²/2 there, so we
    subtract two near-equal quantities.  The Taylor branch
    1/2 + z/6 + z²/24 + z³/120 + z⁴/720 is accurate to ~16 digits for
    |z| < 5e-2 and we keep a generous threshold (1e-2) to stay well clear
    of the cancellation zone.
    """
    import numpy as _np
    z = _np.asarray(z)
    small = _np.abs(z) < 1e-2
    z_safe = _np.where(small, 1.0, z)
    direct = (_np.expm1(z) - z) / (z_safe * z_safe)
    taylor = 0.5 + z * (1.0/6.0
                        + z * (1.0/24.0
                               + z * (1.0/120.0
                                      + z * (1.0/720.0))))
    return _np.where(small, taylor, direct)


@partial(jax.jit, static_argnames=('nz',))
def _lhs_jac_banded_kernel(y, M, k, c0,
                           dzi, Kzz, Dzz, vz, alpha, Tco, ms, g, Ti, Hpi,
                           gas_mask, bot_vdep, use_botflux_flag,
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
    Dzz       : (nz-1, ni)      molecular diffusion (per species)
    vz        : (nz-1,)         vertical velocity at half-levels
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
    vz_pos = jnp.maximum(vz, 0.0)                 # (vz>0)*vz
    vz_neg = jnp.minimum(vz, 0.0)                 # (vz<0)*vz

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

    term_j = Dj * (-1. / Hpi[j][:, None]
                   + ms * g[j][:, None] / (Navo * kb * Ti[j][:, None])
                   + alpha * dTj[:, None] / Ti[j][:, None])
    term_j1 = Dj1 * (-1. / Hpi[j - 1][:, None]
                     + ms * g[j][:, None] / (Navo * kb * Ti[j - 1][:, None])
                     + alpha * dTj1[:, None] / Ti[j - 1][:, None])

    md_d_sc = (-inv_dza[:, None]
               * (Dj / dzi[j][:, None] * (ysum[j + 1] + ysum[j])[:, None] / 2.
                  + Dj1 / dzi[j - 1][:, None] * (ysum[j - 1] + ysum[j])[:, None] / 2.)
               / ysum[j][:, None])
    md_d = md_d_sc + inv_dza2[:, None] * (term_j - term_j1)

    term_u = Dj * (-1. / Hpi[j][:, None]
                   + ms * g[j + 1][:, None] / (Navo * kb * Ti[j][:, None])
                   + alpha * dTj[:, None] / Ti[j][:, None])
    md_u = (inv_dza[:, None] * Dj / dzi[j][:, None]
            * (ysum[j + 1] + ysum[j])[:, None] / (2. * ysum[j + 1][:, None])
            + inv_dza2[:, None] * term_u)

    term_l = Dj1 * (-1. / Hpi[j - 1][:, None]
                    + ms * g[j - 1][:, None] / (Navo * kb * Ti[j - 1][:, None])
                    + alpha * dTj1[:, None] / Ti[j - 1][:, None])
    md_l = (inv_dza[:, None] * Dj1 / dzi[j - 1][:, None]
            * (ysum[j - 1] + ysum[j])[:, None] / (2. * ysum[j - 1][:, None])
            - inv_dza2[:, None] * term_l)

    diag_diff  = diag_diff .at[1:nz - 1].add(-(ek_d[:, None] + md_d))
    upper_diff = upper_diff.at[2:nz    ].add(-(ek_u[:, None] + md_u))
    lower_diff = lower_diff.at[0:nz - 2].add(-(ek_l[:, None] + md_l))

    # ----- bottom BC (j = 0) -----------------------------------------
    mol_bc0 = (-1. / Hpi[0] + ms * g[0] / (Navo * kb * Ti[0])
               + alpha / Ti[0] * (Tco[1] - Tco[0]) / dzi[0])               # (ni,)

    bot0_eddy = (-1. / dzi[0] * (Kzz[0] / dzi[0])
                 * (ysum[1] + ysum[0]) / (2. * ysum[0])
                 - vz_pos[0] / dzi[0])
    bot0_mol  = (-1. / dzi[0] * (Dzz[0] / dzi[0])
                 * (ysum[1] + ysum[0]) / (2. * ysum[0])
                 + 1. / dzi[0] * Dzz[0] / 2. * mol_bc0)                    # (ni,)
    diag_diff = diag_diff.at[0].add(-(bot0_eddy + bot0_mol))
    # use_botflux: ab_diag[0] -= -bot_vdep / dzi[0]
    diag_diff = diag_diff.at[0].add(use_botflux_flag * bot_vdep / dzi[0])

    bot1_eddy = (1. / dzi[0] * (Kzz[0] / dzi[0])
                 * (ysum[1] + ysum[0]) / (2. * ysum[1])
                 - vz_neg[0] / dzi[0])
    bot1_mol  = (1. / dzi[0] * (Dzz[0] / dzi[0])
                 * (ysum[1] + ysum[0]) / (2. * ysum[1])
                 + 1. / dzi[0] * Dzz[0] / 2. * mol_bc0)                    # (ni,)
    upper_diff = upper_diff.at[1].add(-(bot1_eddy + bot1_mol))

    # ----- top BC (j = nz-1) -----------------------------------------
    mol_bcN = (-1. / Hpi[-1] + ms * g[-1] / (Navo * kb * Ti[-1])
               + alpha / Ti[-1] * (Tco[-1] - Tco[-2]) / dzi[-1])           # (ni,)

    topN_eddy = (-1. / dzi[nz - 2] * (Kzz[nz - 2] / dzi[nz - 2])
                 * (ysum[nz - 2] + ysum[nz - 1]) / (2. * ysum[nz - 1])
                 + vz_neg[-1] / dzi[-1])
    topN_mol  = (-1. / dzi[nz - 2] * (Dzz[nz - 2] / dzi[nz - 2])
                 * (ysum[nz - 1] + ysum[nz - 2]) / (2. * ysum[nz - 1])
                 - 1. / dzi[-1] * Dzz[-1] / 2. * mol_bcN)                  # (ni,)
    diag_diff = diag_diff.at[nz - 1].add(-(topN_eddy + topN_mol))

    topL_eddy = (1. / dzi[nz - 2] * (Kzz[nz - 2] / dzi[nz - 2])
                 * (ysum[nz - 2] + ysum[nz - 1]) / (2. * ysum[nz - 2])
                 + vz_pos[-1] / dzi[-1])
    topL_mol  = (1. / dzi[nz - 2] * (Dzz[nz - 2] / dzi[nz - 2])
                 * (ysum[nz - 1] + ysum[nz - 2]) / (2. * ysum[nz - 2])
                 - 1. / dzi[-1] * Dzz[-1] / 2. * mol_bcN)                  # (ni,)
    lower_diff = lower_diff.at[nz - 2].add(-(topL_eddy + topL_mol))

    # ------------------------------------------------------------------
    # 4. Add diffusion contributions to the three active banded rows
    # ------------------------------------------------------------------
    ab = ab.at[bw     ].add(diag_diff .reshape(-1))
    ab = ab.at[bw - ni].add(upper_diff.reshape(-1))
    ab = ab.at[bw + ni].add(lower_diff.reshape(-1))

    return ab


# ===========================================================================
# Log-space banded LHS kernel
#
# DORMANT INFRASTRUCTURE (see note above on _chemdf_logy_single).  Validated
# at machine precision against the manual chain-rule transform of the
# linear-y banded LHS.  Currently not wired into the integration loop.
#
# Same overall structure as `_lhs_jac_banded_kernel`, but:
#   * Chemistry block via `_jac_logy_vmap` (autodiff handles chain rule and
#     the -f_chem/y diagonal correction for free).
#   * Diffusion entries computed in linear y, then transformed by the
#     chain rule:
#       - diag (j' = j):  J_x = J_y - (diff/y)        → add diff_logy on diag
#       - upper (j' = j+1): J_x = J_y * y[j+1] / y[j]
#       - lower (j' = j-1): J_x = J_y * y[j-1] / y[j]
#   * Caller passes `diff_logy = diffdf(exp(x))/exp(x)` (NumPy-side) for the
#     diagonal correction.
# ===========================================================================

@partial(jax.jit, static_argnames=('nz',))
def _lhs_jac_banded_logy_kernel(x, M, k, c0, diff_logy,
                                dzi, Kzz, Dzz, vz, alpha, Tco, ms, g, Ti, Hpi,
                                gas_mask, bot_vdep, use_botflux_flag,
                                nz):
    """Banded LHS = c0*I - dg/dx where g(x) = (chemdf + diffdf)(exp(x)) / exp(x).

    Inputs mirror :func:`_lhs_jac_banded_kernel` with these substitutions:
      y  → x = log(y)
      additionally `diff_logy` (nz, ni) = diffdf(exp(x), atm) / exp(x)
                   computed by the caller, used for the diag chain-rule
                   correction.
    """
    bw = 2 * ni - 1
    y  = jnp.exp(x)

    # ------------------------------------------------------------------
    # 1. Chemistry block via log-space autodiff (chain rule already
    #    applied; diagonal includes -f_chem/y from the quotient rule).
    # ------------------------------------------------------------------
    jac = _jac_logy_vmap(x, M, k)

    ab = jnp.zeros((2 * bw + 1, ni * nz))
    si, sj = jnp.mgrid[0:ni, 0:ni]
    row_chem = bw + si - sj
    col_chem = jnp.arange(nz)[:, None, None] * ni + sj
    ab = ab.at[row_chem[None], col_chem].set(-jac)

    # ------------------------------------------------------------------
    # 2. Identity: c0 on the main diagonal.
    # ------------------------------------------------------------------
    ab = ab.at[bw].add(c0)

    # ------------------------------------------------------------------
    # 3. Diffusion contributions in linear y (same analytic formulas as
    #    the linear kernel).  These will be transformed in step 4.
    # ------------------------------------------------------------------
    ysum   = (y * gas_mask).sum(axis=1)
    vz_pos = jnp.maximum(vz, 0.0)
    vz_neg = jnp.minimum(vz, 0.0)

    diag_diff  = jnp.zeros((nz, ni))
    upper_diff = jnp.zeros((nz, ni))
    lower_diff = jnp.zeros((nz, ni))

    # ----- middle layers j = 1..nz-2 ---------------------------------
    j      = jnp.arange(1, nz - 1)
    dz_ave = 0.5 * (dzi[j - 1] + dzi[j])
    Dj     = Dzz[j]
    Dj1    = Dzz[j - 1]

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

    inv_dza  = 1. / dz_ave
    inv_dza2 = inv_dza / 2.
    dTj      = (Tco[j + 1] - Tco[j]) / dzi[j]
    dTj1     = (Tco[j]     - Tco[j - 1]) / dzi[j - 1]

    term_j = Dj * (-1. / Hpi[j][:, None]
                   + ms * g[j][:, None] / (Navo * kb * Ti[j][:, None])
                   + alpha * dTj[:, None] / Ti[j][:, None])
    term_j1 = Dj1 * (-1. / Hpi[j - 1][:, None]
                     + ms * g[j][:, None] / (Navo * kb * Ti[j - 1][:, None])
                     + alpha * dTj1[:, None] / Ti[j - 1][:, None])

    md_d_sc = (-inv_dza[:, None]
               * (Dj / dzi[j][:, None] * (ysum[j + 1] + ysum[j])[:, None] / 2.
                  + Dj1 / dzi[j - 1][:, None] * (ysum[j - 1] + ysum[j])[:, None] / 2.)
               / ysum[j][:, None])
    md_d = md_d_sc + inv_dza2[:, None] * (term_j - term_j1)

    term_u = Dj * (-1. / Hpi[j][:, None]
                   + ms * g[j + 1][:, None] / (Navo * kb * Ti[j][:, None])
                   + alpha * dTj[:, None] / Ti[j][:, None])
    md_u = (inv_dza[:, None] * Dj / dzi[j][:, None]
            * (ysum[j + 1] + ysum[j])[:, None] / (2. * ysum[j + 1][:, None])
            + inv_dza2[:, None] * term_u)

    term_l = Dj1 * (-1. / Hpi[j - 1][:, None]
                    + ms * g[j - 1][:, None] / (Navo * kb * Ti[j - 1][:, None])
                    + alpha * dTj1[:, None] / Ti[j - 1][:, None])
    md_l = (inv_dza[:, None] * Dj1 / dzi[j - 1][:, None]
            * (ysum[j - 1] + ysum[j])[:, None] / (2. * ysum[j - 1][:, None])
            - inv_dza2[:, None] * term_l)

    diag_diff  = diag_diff .at[1:nz - 1].add(-(ek_d[:, None] + md_d))
    upper_diff = upper_diff.at[2:nz    ].add(-(ek_u[:, None] + md_u))
    lower_diff = lower_diff.at[0:nz - 2].add(-(ek_l[:, None] + md_l))

    # ----- bottom BC (j = 0) -----------------------------------------
    mol_bc0 = (-1. / Hpi[0] + ms * g[0] / (Navo * kb * Ti[0])
               + alpha / Ti[0] * (Tco[1] - Tco[0]) / dzi[0])

    bot0_eddy = (-1. / dzi[0] * (Kzz[0] / dzi[0])
                 * (ysum[1] + ysum[0]) / (2. * ysum[0])
                 - vz_pos[0] / dzi[0])
    bot0_mol  = (-1. / dzi[0] * (Dzz[0] / dzi[0])
                 * (ysum[1] + ysum[0]) / (2. * ysum[0])
                 + 1. / dzi[0] * Dzz[0] / 2. * mol_bc0)
    diag_diff = diag_diff.at[0].add(-(bot0_eddy + bot0_mol))
    # bot_vdep: included exactly as in the linear kernel.  Its contribution
    # to g(x) = diff/y is -v_dep/dz (constant in x), so the linear diagonal
    # term and the +diff_logy correction below cancel each other exactly.
    diag_diff = diag_diff.at[0].add(use_botflux_flag * bot_vdep / dzi[0])

    bot1_eddy = (1. / dzi[0] * (Kzz[0] / dzi[0])
                 * (ysum[1] + ysum[0]) / (2. * ysum[1])
                 - vz_neg[0] / dzi[0])
    bot1_mol  = (1. / dzi[0] * (Dzz[0] / dzi[0])
                 * (ysum[1] + ysum[0]) / (2. * ysum[1])
                 + 1. / dzi[0] * Dzz[0] / 2. * mol_bc0)
    upper_diff = upper_diff.at[1].add(-(bot1_eddy + bot1_mol))

    # ----- top BC (j = nz-1) -----------------------------------------
    mol_bcN = (-1. / Hpi[-1] + ms * g[-1] / (Navo * kb * Ti[-1])
               + alpha / Ti[-1] * (Tco[-1] - Tco[-2]) / dzi[-1])

    topN_eddy = (-1. / dzi[nz - 2] * (Kzz[nz - 2] / dzi[nz - 2])
                 * (ysum[nz - 2] + ysum[nz - 1]) / (2. * ysum[nz - 1])
                 + vz_neg[-1] / dzi[-1])
    topN_mol  = (-1. / dzi[nz - 2] * (Dzz[nz - 2] / dzi[nz - 2])
                 * (ysum[nz - 1] + ysum[nz - 2]) / (2. * ysum[nz - 1])
                 - 1. / dzi[-1] * Dzz[-1] / 2. * mol_bcN)
    diag_diff = diag_diff.at[nz - 1].add(-(topN_eddy + topN_mol))

    topL_eddy = (1. / dzi[nz - 2] * (Kzz[nz - 2] / dzi[nz - 2])
                 * (ysum[nz - 2] + ysum[nz - 1]) / (2. * ysum[nz - 2])
                 + vz_pos[-1] / dzi[-1])
    topL_mol  = (1. / dzi[nz - 2] * (Dzz[nz - 2] / dzi[nz - 2])
                 * (ysum[nz - 1] + ysum[nz - 2]) / (2. * ysum[nz - 2])
                 - 1. / dzi[-1] * Dzz[-1] / 2. * mol_bcN)
    lower_diff = lower_diff.at[nz - 2].add(-(topL_eddy + topL_mol))

    # ------------------------------------------------------------------
    # 4. Chain-rule transform of the diffusion bands.
    #
    #    diag_diff is -J_y[diag] (per linear assembly).  In log-space:
    #      -J_x[diag] = -(J_y[diag] - g_diff[diag]) = -J_y[diag] + g_diff
    #    so we ADD diff_logy to diag_diff.
    #
    #    upper_diff at row k = j+1 currently holds -J_y[i^j, i^{j+1}].
    #    In log-space: -J_x = -J_y * y[j+1]/y[j], so multiply by y[k]/y[k-1].
    #
    #    lower_diff at row k = j-1 currently holds -J_y[i^j, i^{j-1}].
    #    In log-space: multiply by y[k]/y[k+1].
    # ------------------------------------------------------------------
    ratio_upper = y[1:] / y[:-1]            # (nz-1, ni): ratio for k = 1..nz-1
    ratio_lower = y[:-1] / y[1:]            # (nz-1, ni): ratio for k = 0..nz-2

    upper_diff = upper_diff.at[1:nz ].set(upper_diff[1:nz ] * ratio_upper)
    lower_diff = lower_diff.at[0:nz-1].set(lower_diff[0:nz-1] * ratio_lower)

    # Diagonal correction: ADD diff_logy (this absorbs -(-g_diff) and also
    # absorbs the bot_vdep cancellation we deliberately omitted above).
    diag_diff = diag_diff + diff_logy

    # ------------------------------------------------------------------
    # 5. Add transformed diffusion contributions to the three active bands.
    # ------------------------------------------------------------------
    ab = ab.at[bw     ].add(diag_diff .reshape(-1))
    ab = ab.at[bw - ni].add(upper_diff.reshape(-1))
    ab = ab.at[bw + ni].add(lower_diff.reshape(-1))

    return ab
