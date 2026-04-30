#!/usr/bin/env python3
"""
profile_wallclock.py — wall-clock breakdown of a neoVULCAN run.

Creates a temp copy of neoVULCAN, patches ode_solver.py with inline
perf_counter calls around every major operation in solver(), runs the
simulation for a configurable number of steps, and prints a breakdown.

Usage:
    python profile_wallclock.py [--steps N]   (default: 500)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

HERE   = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Config overrides (same as test_regression.py)
# ---------------------------------------------------------------------------

def cfg_overrides(n_steps: int) -> str:
    return textwrap.dedent(f"""\
        # ---- profile_wallclock.py overrides ----
        count_max       = {n_steps}
        runtime         = 1e30
        use_live_plot   = False
        use_plot_end    = False
        use_plot_evo    = False
        use_save_movie  = False
        use_flux_movie  = False
        save_evolution  = False
        out_name        = 'profile_output.vul'
        # ini_mix         = 'const_mix'
        # const_mix       = {{'H2': 0.855, 'He': 0.144, 'H2O': 5e-4, 'PH3': 6e-7}}
        use_ini_cold_trap = False
    """)


# ---------------------------------------------------------------------------
# Patch for ode_solver.py — instruments the banded solver() with inline
# perf_counter calls and prints a summary at exit (via atexit).
#
# The banded solver() call sequence (default config, use_lhs_jac_banded=True):
#   A:  df = chemdf(y,M,k) + diffdf(y, atm)          ← first RHS
#       if use_banded:
#   B:      lhs_b, bw = jac_fn(var, atm)              ← Jacobian (banded)
#   C:  k1_flat = solve_banded(...)                   ← k1 solve (no store_bandM)
#       k1 = k1_flat.reshape(...)
#   D:  df = chemdf(yk2,M,k) + diffdf(yk2, atm)      ← second RHS
#   E:  k2 = solve_banded(...)                        ← k2 solve
# ---------------------------------------------------------------------------

_PATCH_HEADER = '''\
import atexit as _atexit, time as _time
_T = {k: 0.0 for k in ['chemdf','diffdf','neg_achemjac','diff_jac',
                         'store_bandM','solve_banded','other_jac','photo']}
_N = [0]  # step counter

def _print_timing():
    n = max(_N[0], 1)
    total = sum(_T.values())
    print()
    print("=" * 62)
    print(f" Wall-clock breakdown  ({n} steps,  {total/n*1e3:.2f} ms/step total)")
    print("=" * 62)
    print(f"  {'Operation':<30} {'total':>8}  {'per step':>9}  {'%':>5}")
    print("  " + "-" * 58)
    for k in ['chemdf','diffdf','neg_achemjac','diff_jac',
              'store_bandM','solve_banded','other_jac','photo']:
        pct = 100*_T[k]/total if total else 0
        print(f"  {k:<30} {_T[k]:>7.2f}s  {_T[k]/n*1e3:>8.2f}ms  {pct:>5.1f}%")
    print("  " + "-" * 58)
    print(f"  {'TOTAL':<30} {total:>7.2f}s  {total/n*1e3:>8.2f}ms")
    print()

_atexit.register(_print_timing)
'''

# Patch A+B: split first chemdf/diffdf; time banded Jacobian call.
# Unique because the non-banded solver variant has `lhs = jac_tot(var, atm)`
# immediately after `df = chemdf...`, not `if use_banded:`.
_OLD_SOLVER_CORE = (
    '        df = chemdf(y,M,k).flatten() + diffdf(y, atm).flatten()\n'
    '\n'
    '        if use_banded:\n'
    '            lhs_b, bw = jac_fn(var, atm)\n'
)

_NEW_SOLVER_CORE = (
    "        _t0 = _time.perf_counter()\n"
    "        _chem1 = chemdf(y,M,k).flatten()\n"
    "        _T['chemdf'] += _time.perf_counter() - _t0\n"
    "\n"
    "        _t0 = _time.perf_counter()\n"
    "        _diff1 = diffdf(y, atm).flatten()\n"
    "        _T['diffdf'] += _time.perf_counter() - _t0\n"
    "\n"
    "        df = _chem1 + _diff1\n"
    "\n"
    "        if use_banded:\n"
    "            _t0 = _time.perf_counter()\n"
    "            lhs_b, bw = jac_fn(var, atm)\n"
    "            _T['neg_achemjac'] += _time.perf_counter() - _t0\n"
)

# Patch C: time k1 solve.
# Unique because in the non-banded variant there is a blank line between
# `k1_flat = solve_banded(...)` and `k1 = k1_flat.reshape(...)`.
_OLD_K1SOLVE = (
    '        k1_flat = scipy.linalg.solve_banded((bw,bw),lhs_b,df)\n'
    '        k1 = k1_flat.reshape(y.shape)\n'
)

_NEW_K1SOLVE = (
    "        _t0 = _time.perf_counter()\n"
    "        k1_flat = scipy.linalg.solve_banded((bw,bw),lhs_b,df)\n"
    "        _T['solve_banded'] += _time.perf_counter() - _t0\n"
    "        k1 = k1_flat.reshape(y.shape)\n"
)

# Patch D: split second chemdf/diffdf.
# Two occurrences exist; replace(..., 1) hits the banded-path one first.
_OLD_CHEM2 = (
    '        df = chemdf(yk2,M,k).flatten() + diffdf(yk2, atm).flatten()\n'
)

_NEW_CHEM2 = (
    "        _t0 = _time.perf_counter()\n"
    "        _chem2 = chemdf(yk2,M,k).flatten()\n"
    "        _T['chemdf'] += _time.perf_counter() - _t0\n"
    "\n"
    "        _t0 = _time.perf_counter()\n"
    "        _diff2 = diffdf(yk2, atm).flatten()\n"
    "        _T['diffdf'] += _time.perf_counter() - _t0\n"
    "\n"
    "        df = _chem2 + _diff2\n"
)

# Patch E: time k2 solve and count the completed solver() call.
# Two occurrences exist; replace(..., 1) hits the banded-path one first.
_OLD_BANDSOL2 = (
    '        k2 = scipy.linalg.solve_banded((bw,bw),lhs_b,rhs)\n'
)

_NEW_BANDSOL2 = (
    "        _t0 = _time.perf_counter()\n"
    "        k2 = scipy.linalg.solve_banded((bw,bw),lhs_b,rhs)\n"
    "        _T['solve_banded'] += _time.perf_counter() - _t0\n"
    "        _N[0] += 1\n"
)


def apply_patches(src: str) -> str:
    """Apply all timing patches to ode_solver.py source."""
    patches = [
        (_OLD_SOLVER_CORE, _NEW_SOLVER_CORE),
        (_OLD_K1SOLVE,     _NEW_K1SOLVE),
        (_OLD_CHEM2,       _NEW_CHEM2),
        (_OLD_BANDSOL2,    _NEW_BANDSOL2),
    ]
    for old, new in patches:
        if old not in src:
            raise RuntimeError(
                f'Patch anchor not found in ode_solver.py:\n{old!r}'
            )
        src = src.replace(old, new, 1)

    # Prepend the timer header right after the last top-level import.
    lines = src.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_at = i + 1
    lines.insert(insert_at, _PATCH_HEADER)
    return ''.join(lines)


def run(n_steps: int):
    with tempfile.TemporaryDirectory(prefix='vulcan_wallclock_') as tmp:
        shutil.copytree(HERE, tmp, dirs_exist_ok=True)

        # Apply test config
        with open(os.path.join(tmp, 'vulcan_cfg.py'), 'a') as f:
            f.write('\n')
            f.write(cfg_overrides(n_steps))

        # Patch ode_solver.py
        solver_path = os.path.join(tmp, 'src', 'ode_solver.py')
        with open(solver_path) as f:
            src = f.read()
        with open(solver_path, 'w') as f:
            f.write(apply_patches(src))

        print(f"Running {n_steps} steps with wall-clock instrumentation …\n")
        result = subprocess.run(
            [PYTHON, 'vulcan.py', '-n'],
            cwd=tmp, capture_output=False, text=True, timeout=1800,
        )
        if result.returncode != 0:
            raise RuntimeError(f'vulcan.py exited with code {result.returncode}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500)
    args = parser.parse_args()
    run(args.steps)


if __name__ == '__main__':
    main()
