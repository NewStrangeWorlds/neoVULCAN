#!/usr/bin/env python3
"""
test_diffdf.py — validate refactored diffdf methods against a golden baseline.

Workflow
--------
Step 1 (BEFORE any refactoring): record the baseline outputs
    python test_diffdf.py --save

Step 2 (AFTER each method is refactored): compare against baseline
    python test_diffdf.py

The script tests all five diffdf variants under three boundary-condition
combinations (no flux / top flux only / bot flux only), producing 15 cases.
"""

import argparse
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, 'test_diffdf_baseline.pkl')

# ---------------------------------------------------------------------------
# Path setup — ode_solver lives in src/, which imports vulcan_cfg from HERE
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(HERE, 'src'))
os.chdir(HERE)   # vulcan_cfg and chem_funs must be importable from cwd


# ---------------------------------------------------------------------------
# Synthetic atmosphere generator
# ---------------------------------------------------------------------------

def make_inputs(nz, ni, seed=42):
    """Return (y, atm) with realistic shapes and plausible physical values."""
    rng = np.random.default_rng(seed)

    # Cell-centre temperature, gravity (nz,)
    Tco = 500. + 1500. * rng.random(nz)           # 500–2000 K
    g   = 24.79 * np.ones(nz)                      # Jupiter surface gravity

    # Interface quantities (nz-1,)
    Ti  = 0.5 * (Tco[:-1] + Tco[1:])              # interface temperature
    Hp  = 8.314 * Ti / (g[:-1] * 2.0e-3)          # scale height [m], m_bar=2 g/mol
    Hpi = Hp

    # Grid spacings (nz-1,) — log-spaced 1 bar → 1e-6 bar
    P   = np.logspace(0, -6, nz)
    dz  = 8.314 * Ti / (g[:-1] * 2.0e-3) * np.log(P[:-1] / P[1:])
    dzi = dz                                        # alias used in ode_solver

    # Eddy diffusion and vertical wind (nz-1,)
    Kzz = 1e5 * np.ones(nz - 1)                   # cm²/s
    vz  = 1e-3 * rng.standard_normal(nz - 1)       # cm/s, small

    # Molecular masses (ni,) and thermal diffusion factors (ni,)
    ms    = 1.67e-24 * (2. + 28. * rng.random(ni))  # 2–30 amu in grams
    alpha = 0.1 * rng.standard_normal(ni)

    # Molecular diffusion coefficients at interfaces (nz-1, ni)
    Dzz = 1e4 * np.abs(rng.standard_normal((nz - 1, ni))) + 1.

    # Mean molecular velocity at cell centres (nz, ni) — used by diffdf_vm
    vm = 1e-2 * rng.standard_normal((nz, ni))

    # Settling velocity at cell centres (nz, ni) — used by diffdf_settling
    vs = -1e-3 * np.abs(rng.standard_normal((nz, ni)))  # negative = downward

    # Mixing ratios (nz, ni) — positive, sum to ~1
    raw = np.abs(rng.standard_normal((nz, ni))) + 1e-10
    y   = raw / raw.sum(axis=1, keepdims=True)

    # Flux boundary conditions (ni,)
    top_flux = 1e6 * rng.standard_normal(ni)
    bot_flux = 1e5 * rng.standard_normal(ni)
    bot_vdep = np.abs(1e-2 * rng.standard_normal(ni))

    atm = SimpleNamespace(
        dzi=dzi, Kzz=Kzz, vz=vz,
        Dzz=Dzz, alpha=alpha, Tco=Tco,
        ms=ms, g=g, Ti=Ti, Hpi=Hpi, Hp=Hp,
        vm=vm, vs=vs,
        top_flux=top_flux, bot_flux=bot_flux, bot_vdep=bot_vdep,
        gas_indx=np.arange(ni),      # all species are gaseous
    )
    return y, atm


# ---------------------------------------------------------------------------
# Build test cases: (method_name, bc_label, cfg_overrides)
# ---------------------------------------------------------------------------

METHODS = [
    'diffdf_no_mol',
    'diffdf',
    'diffdf_vm',
    'diffdf_settling',
    'diffdf_settling_vm',
]

BC_CASES = [
    ('no_flux',  dict(use_topflux=False, use_botflux=False)),
    ('top_flux', dict(use_topflux=True,  use_botflux=False)),
    ('bot_flux', dict(use_topflux=False, use_botflux=True)),
]


def run_all(solver, y, atm, vulcan_cfg):
    results = {}
    for method_name in METHODS:
        method = getattr(solver, method_name)
        for bc_label, bc_flags in BC_CASES:
            for attr, val in bc_flags.items():
                setattr(vulcan_cfg, attr, val)
            key = f'{method_name}/{bc_label}'
            results[key] = method(y, atm).copy()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true',
                        help='Record golden baseline from current code')
    args = parser.parse_args()

    # Imports that depend on the working directory being set correctly
    import vulcan_cfg
    import chem_funs
    from chem_funs import ni
    from vulcan_cfg import nz
    from ode_solver import ODESolver

    # Disable flags that would cause import-time side effects
    vulcan_cfg.non_gas_sp = None
    vulcan_cfg.use_condense = False
    vulcan_cfg.use_fix_sp_bot = {}

    y, atm = make_inputs(nz, ni)
    solver = ODESolver()

    if args.save:
        results = run_all(solver, y, atm, vulcan_cfg)
        with open(BASELINE, 'wb') as f:
            pickle.dump(results, f)
        print(f'Baseline saved → {BASELINE}')
        print(f'  {len(results)} cases recorded.')
        return

    # Validate mode
    if not os.path.exists(BASELINE):
        sys.exit(f'No baseline found at {BASELINE}.\nRun with --save first.')

    with open(BASELINE, 'rb') as f:
        golden = pickle.load(f)

    results = run_all(solver, y, atm, vulcan_cfg)

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
