#!/usr/bin/env python3
"""
test_lhsjac.py — validate refactored lhs_jac methods against a golden baseline.

Workflow
--------
Step 1 (BEFORE any refactoring): record the baseline outputs
    python test_lhsjac.py --save

Step 2 (AFTER each method is refactored): compare against baseline
    python test_lhsjac.py

Tests the six live lhs_jac variants (lhs_jac_banded is skipped — it is already
vectorised and not being refactored):
    lhs_jac_tot_vm
    lhs_jac_no_mol
    lhs_jac_fix_all_bot
    lhs_jac_no_mol_fix_all_bot
    lhs_jac_settling
    lhs_jac_settling_vm

Each method is exercised with two boundary-condition combinations:
    no_botflux  — use_botflux=False
    with_botflux — use_botflux=True

(diff_esc is set to [] for all cases to avoid needing real species names.)

var.k = {} causes neg_achemjac to use all-zero rates, so the chemistry
Jacobian contribution cancels and the test purely validates the diffusion
coefficient arithmetic.
"""

import argparse
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, 'test_lhsjac_baseline.pkl')

sys.path.insert(0, os.path.join(HERE, 'src'))
os.chdir(HERE)


def make_inputs(nz, ni, seed=42):
    """Return (var, atm) with realistic shapes and plausible physical values."""
    rng = np.random.default_rng(seed)

    Tco = 500. + 1500. * rng.random(nz)
    g   = 24.79 * np.ones(nz)
    Ti  = 0.5 * (Tco[:-1] + Tco[1:])
    Hp  = 8.314 * Ti / (g[:-1] * 2.0e-3)
    Hpi = Hp

    P   = np.logspace(0, -6, nz)
    dz  = 8.314 * Ti / (g[:-1] * 2.0e-3) * np.log(P[:-1] / P[1:])
    dzi = dz

    Kzz = 1e5 * np.ones(nz - 1)
    vz  = 1e-3 * rng.standard_normal(nz - 1)

    ms    = 1.67e-24 * (2. + 28. * rng.random(ni))
    alpha = 0.1 * rng.standard_normal(ni)
    mu    = 2.0 * np.ones(nz)

    Dzz = 1e4 * np.abs(rng.standard_normal((nz - 1, ni))) + 1.
    vm  = 1e-2 * rng.standard_normal((nz, ni))
    vs  = -1e-3 * np.abs(rng.standard_normal((nz, ni)))

    raw = np.abs(rng.standard_normal((nz, ni))) + 1e-10
    y   = raw / raw.sum(axis=1, keepdims=True)

    top_flux = 1e6 * rng.standard_normal(ni)
    bot_flux = 1e5 * rng.standard_normal(ni)
    bot_vdep = np.abs(1e-2 * rng.standard_normal(ni))

    # M: total number density (nz,), order-of-magnitude realistic
    M = 1e19 * np.exp(-np.linspace(0, 10, nz))

    atm = SimpleNamespace(
        dzi=dzi, Kzz=Kzz, vz=vz,
        Dzz=Dzz, alpha=alpha, Tco=Tco,
        ms=ms, mu=mu, g=g, Ti=Ti, Hpi=Hpi, Hp=Hp,
        vm=vm, vs=vs, M=M,
        top_flux=top_flux, bot_flux=bot_flux, bot_vdep=bot_vdep,
        gas_indx=np.arange(ni),
    )

    var = SimpleNamespace(
        y=y,
        dt=1.0,
        k={},    # zero chemistry rates → neg_achemjac returns zeros
        ymix=y.copy(),
    )

    return var, atm


METHODS = [
    'lhs_jac_tot_vm',
    'lhs_jac_no_mol',
    'lhs_jac_fix_all_bot',
    'lhs_jac_no_mol_fix_all_bot',
    'lhs_jac_settling',
    'lhs_jac_settling_vm',
]

BC_CASES = [
    ('no_botflux',   dict(use_botflux=False)),
    ('with_botflux', dict(use_botflux=True)),
]


def run_all(solver, var, atm, vulcan_cfg):
    results = {}
    for method_name in METHODS:
        method = getattr(solver, method_name)
        for bc_label, bc_flags in BC_CASES:
            for attr, val in bc_flags.items():
                setattr(vulcan_cfg, attr, val)
            key = f'{method_name}/{bc_label}'
            results[key] = method(var, atm).copy()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true',
                        help='Record golden baseline from current code')
    args = parser.parse_args()

    import vulcan_cfg
    import chem_funs
    from chem_funs import ni, nr
    from vulcan_cfg import nz
    from ode_solver import ODESolver

    vulcan_cfg.non_gas_sp   = None
    vulcan_cfg.use_condense = False
    vulcan_cfg.use_fix_sp_bot = {}
    vulcan_cfg.diff_esc     = []

    var, atm = make_inputs(nz, ni)
    # Provide zero rate constants so neg_achemjac returns a zero matrix;
    # this isolates the diffusion terms in the test.
    var.k = {i: np.zeros(nz) for i in range(1, nr + 1)}
    solver = ODESolver()

    if args.save:
        results = run_all(solver, var, atm, vulcan_cfg)
        with open(BASELINE, 'wb') as f:
            pickle.dump(results, f)
        print(f'Baseline saved → {BASELINE}')
        print(f'  {len(results)} cases recorded.')
        return

    if not os.path.exists(BASELINE):
        sys.exit(f'No baseline found at {BASELINE}.\nRun with --save first.')

    with open(BASELINE, 'rb') as f:
        golden = pickle.load(f)

    results = run_all(solver, var, atm, vulcan_cfg)

    passed = failed = 0
    for key in sorted(golden):
        ref = golden[key]
        got = results.get(key)
        if got is None:
            print(f'  MISSING  {key}')
            failed += 1
            continue
        if np.allclose(ref, got, rtol=1e-12, atol=0):
            print(f'  OK       {key}')
            passed += 1
        else:
            diff = np.abs(ref - got)
            rel  = diff / (np.abs(ref) + 1e-300)
            print(f'  FAIL     {key}  max_abs={diff.max():.2e}  max_rel={rel.max():.2e}')
            failed += 1

    print(f'\n{passed} passed, {failed} failed out of {passed+failed} cases.')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
