#!/usr/bin/env python
"""Validate the fused JAX `_lhs_jac_banded_kernel` against the legacy NumPy
assembly variants element-wise.

Covers all four variants that previously used numpy assembly:
  - banded (no settling, no vm)              vs lhs_jac_banded_numpy
  - settling (use_settling=True)             vs lhs_jac_settling + store_bandM
  - vm-advection (use_vm_mol=True)           vs lhs_jac_tot_vm + store_bandM
  - settling + vm                             vs lhs_jac_settling_vm + store_bandM
  - no mol-diff (use_moldiff=False)          vs lhs_jac_no_mol + store_bandM

Each variant is run on the Earth-like setup with the matching cfg flags so the
JAX cache uses the right vs/vm/Dzz/thermal_flag/vm_bot_flag values.

Run from /home/kitzmann/Code/VULCAN/neoVULCAN.
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"

_base = os.path.dirname(os.path.abspath(__file__))
_src  = os.path.join(_base, 'src')
for _p in (_base, _src):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.chdir(_base)

import vulcan_cfg
# Use a constant-mix init to avoid depending on stellar/photo tables.
vulcan_cfg.ini_mix   = 'const_mix'
vulcan_cfg.use_photo = False

import numpy as np

# Build chemistry_jax.py if missing/stale (quiet).
python_executable = sys.executable
os.system(python_executable + ' make_chemistry_jax.py > /dev/null 2>&1')

import store, build_atm
import chemistry_jax as chem_funs
from chemistry_jax import ni
from rates import ReadRate
from ros2 import Ros2
from vulcan_cfg import nz


def setup():
    data_var  = store.Variables()
    data_atm  = store.AtmData()
    data_para = store.Parameters()

    make_atm = build_atm.Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)

    rate = ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    if vulcan_cfg.use_lowT_limit_rates:
        data_var = rate.lim_lowT_rates(data_var, data_atm)
    data_var = rate.rev_rate(data_var, data_atm)
    data_var = rate.remove_rate(data_var)

    ini_abun = build_atm.InitialAbun()
    data_var = ini_abun.ini_y(data_var, data_atm)
    data_var = ini_abun.ele_sum(data_var)

    from output import Output
    output = Output()
    data_atm = make_atm.f_mu_dz(data_var, data_atm, output)
    make_atm.mol_diff(data_atm)
    make_atm.BC_flux(data_atm)

    # Seed settling velocities (non-zero) so the test exercises the vs path.
    # Use a synthetic Stokes-like field so values are physically reasonable.
    rng = np.random.default_rng(42)
    data_atm.vs = rng.normal(0, 0.01, size=(nz - 1, ni))
    # Seed vm (cell-centered) with a synthetic field for the vm tests.
    data_atm.vm = rng.normal(0, 0.01, size=(nz, ni))

    solver = Ros2()
    solver.naming_solver(data_para)
    return solver, data_var, data_atm


def run_case(label, cfg_flags, solver, data_var, data_atm, numpy_method,
             numpy_returns_banded=False):
    """Compute Path A (fused JAX kernel) and Path B (numpy assembly + store_bandM)
    for the given cfg flag combination and report the discrepancy.

    Set numpy_returns_banded=True for reference methods that already return
    (ab, bw) (e.g. lhs_jac_banded_numpy) — they skip store_bandM.
    """
    saved = {k: getattr(vulcan_cfg, k, None) for k in cfg_flags}
    for k, v in cfg_flags.items():
        setattr(vulcan_cfg, k, v)
    try:
        # Re-build the JAX cache so it picks up the new flags.
        solver.invalidate_atm_cache()

        # Path A: fused JAX kernel.  lhs_jac_banded now returns the matrix in
        # LAPACK band storage (3*bw+1 rows; top bw rows are dgbtrf workspace).
        # Slice the bottom 2*bw+1 rows to recover scipy solve_banded format
        # for an apples-to-apples comparison with the numpy reference.
        lhs_b_A_lapack, bw_A = solver.lhs_jac_banded(data_var, data_atm)
        lhs_b_A = lhs_b_A_lapack[bw_A:]

        # Path B: numpy assembly → (dense → store_bandM, or directly banded).
        if numpy_returns_banded:
            lhs_b_B, bw_B = getattr(solver, numpy_method)(data_var, data_atm)
        else:
            lhs_dense  = getattr(solver, numpy_method)(data_var, data_atm)
            lhs_b_B, bw_B = solver.store_bandM(lhs_dense, ni, nz)
    finally:
        for k, v in saved.items():
            setattr(vulcan_cfg, k, v)
        solver.invalidate_atm_cache()

    assert bw_A == bw_B, f"{label}: bw mismatch {bw_A} vs {bw_B}"
    diff    = np.abs(lhs_b_A - lhs_b_B)
    max_abs = float(np.max(diff))
    denom   = np.abs(lhs_b_B)
    mask    = denom > 0
    rel     = np.where(mask, diff / np.where(mask, denom, 1.0), 0.0)
    max_rel = float(np.max(rel))
    idx_abs = np.unravel_index(int(np.argmax(diff)), diff.shape)
    idx_rel = np.unravel_index(int(np.argmax(rel)),  rel.shape)
    print(f"[{label:>16}]  max|diff| = {max_abs:.3e}   max rel = {max_rel:.3e}"
          f"   at abs={idx_abs}   rel_at A={lhs_b_A[idx_rel]:.3e}, B={lhs_b_B[idx_rel]:.3e}")
    return max_abs, max_rel


def main():
    solver, data_var, data_atm = setup()

    print("=" * 88)
    print(" Fused JAX kernel vs legacy numpy variants (element-wise)")
    print("=" * 88)

    cases = [
        # label, cfg flag overrides, numpy reference method, numpy_returns_banded
        ("banded (orig)",   {'use_moldiff': True,  'use_settling': False, 'use_vm_mol': False},
                            'lhs_jac_banded_numpy', True),
        ("settling",        {'use_moldiff': True,  'use_settling': True,  'use_vm_mol': False},
                            'lhs_jac_settling', False),
        ("vm",              {'use_moldiff': True,  'use_settling': False, 'use_vm_mol': True},
                            'lhs_jac_tot_vm', False),
        ("settling+vm",     {'use_moldiff': True,  'use_settling': True,  'use_vm_mol': True},
                            'lhs_jac_settling_vm', False),
        ("no_mol",          {'use_moldiff': False, 'use_settling': False, 'use_vm_mol': False},
                            'lhs_jac_no_mol', False),
    ]

    fail = False
    tol_abs = 1e-8     # acceptable absolute diff for the banded matrix
    tol_rel = 1e-10    # acceptable relative diff
    for label, cfg_flags, np_method, banded in cases:
        try:
            max_abs, max_rel = run_case(label, cfg_flags, solver, data_var, data_atm, np_method,
                                        numpy_returns_banded=banded)
        except Exception as e:
            print(f"[{label:>16}]  ERROR: {e}")
            fail = True
            continue
        if max_rel > tol_rel and max_abs > tol_abs:
            print(f"[{label:>16}]  *** EXCEEDS TOLERANCE (rel>{tol_rel:.0e}, abs>{tol_abs:.0e})")
            fail = True

    print("=" * 88)
    if fail:
        print("FAIL")
        sys.exit(1)
    print("PASS")


if __name__ == '__main__':
    main()
