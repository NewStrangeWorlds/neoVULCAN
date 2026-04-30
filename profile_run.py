#!/usr/bin/env python3
"""
profile_run.py — profile a full neoVULCAN run and report where time goes.

Copies neoVULCAN into a temp directory, applies the same config overrides
as test_regression.py, runs vulcan.py under cProfile, then displays a
grouped summary of cumulative time by subsystem.

Usage:
    python profile_run.py [--steps N]   (default: 1000)
"""

import argparse
import os
import pstats
import shutil
import subprocess
import sys
import tempfile
import textwrap

HERE   = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def cfg_overrides(n_steps: int) -> str:
    return textwrap.dedent(f"""\
        # ---- profile_run.py overrides ----
        count_max       = {n_steps}
        runtime         = 1e30
        use_live_plot   = False
        use_plot_end    = False
        use_plot_evo    = False
        use_save_movie  = False
        use_flux_movie  = False
        save_evolution  = False
        out_name        = 'profile_output.vul'
        ini_mix         = 'const_mix'
        const_mix       = {{'H2': 0.855, 'He': 0.144, 'H2O': 5e-4, 'PH3': 6e-7}}
        use_ini_cold_trap = False
    """)


def run_profile(n_steps: int) -> str:
    stats_file = os.path.join(tempfile.gettempdir(), 'neovulcan_profile.stats')

    with tempfile.TemporaryDirectory(prefix='vulcan_profile_') as tmp:
        shutil.copytree(HERE, tmp, dirs_exist_ok=True)

        with open(os.path.join(tmp, 'vulcan_cfg.py'), 'a') as f:
            f.write('\n')
            f.write(cfg_overrides(n_steps))

        cmd = [
            PYTHON, '-m', 'cProfile',
            '-o', stats_file,
            'vulcan.py', '-n',
        ]
        print(f"Running {n_steps} steps under cProfile …")
        result = subprocess.run(
            cmd, cwd=tmp, capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])
            raise RuntimeError(f'vulcan.py exited with code {result.returncode}')

    return stats_file


def _cumtime(st: pstats.Stats) -> dict:
    """Return {func_key: cumtime} for all entries."""
    return {k: v[3] for k, v in st.stats.items()}


def report(stats_file: str, n_steps: int, top_n: int = 25):
    st = pstats.Stats(stats_file)
    st.strip_dirs()

    total = sum(v[2] for v in st.stats.values())  # total tt across all funcs

    # -----------------------------------------------------------------------
    # Subsystem grouping — maps a substring of the file/function name to a label
    # -----------------------------------------------------------------------
    GROUPS = [
        ('chemistry_jax',  'JAX chemistry (chemdf + jacfwd)'),
        ('_jac_jit',       'JAX chemistry (chemdf + jacfwd)'),
        ('chemdf',         'JAX chemistry (chemdf + jacfwd)'),
        ('block_diag',     'Jacobian block-diag assembly'),
        ('solve_banded',   'Banded linear solve (scipy)'),
        ('ode_solver',     'ODE solver (Ros2)'),
        ('one_step',       'ODE solver (Ros2)'),
        ('diffdf',         'Diffusion RHS'),
        ('lhs_jac',        'Diffusion Jacobian'),
        ('compute_tau',    'Radiative transfer'),
        ('compute_flux',   'Radiative transfer'),
        ('compute_J',      'Radiative transfer'),
        ('rates',          'Reaction rates'),
        ('integration',    'Integration loop overhead'),
        ('f_dy',           'Convergence check (f_dy)'),
        ('jax',            'JAX dispatch / XLA overhead'),
        ('numpy',          'NumPy misc'),
    ]

    group_time: dict[str, float] = {}
    seen: set = set()

    for (file, lineno, func), (pc, nc, tt, ct, callers) in st.stats.items():
        key = f'{file}:{func}'
        label = None
        for substr, grp in GROUPS:
            if substr.lower() in file.lower() or substr.lower() in func.lower():
                label = grp
                break
        if label:
            group_time[label] = group_time.get(label, 0.0) + ct
            seen.add(key)

    # -----------------------------------------------------------------------
    # Print grouped summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f" neoVULCAN profile — {n_steps} steps    (total wall: {total:.1f} s)")
    print(f"{'='*65}")
    print(f"{'Subsystem':<42} {'cumtime':>8}   {'%':>5}")
    print(f"{'-'*65}")
    for label, ct in sorted(group_time.items(), key=lambda x: -x[1]):
        pct = 100 * ct / total if total else 0
        print(f"  {label:<40} {ct:>7.2f}s  {pct:>5.1f}%")

    # -----------------------------------------------------------------------
    # Top N raw functions
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f" Top {top_n} functions by cumulative time")
    print(f"{'='*65}")
    print(f"{'Function':<45} {'cumtime':>8}   {'%':>5}")
    print(f"{'-'*65}")

    items = sorted(st.stats.items(), key=lambda x: -x[1][3])
    for (file, lineno, func), (pc, nc, tt, ct, callers) in items[:top_n]:
        pct = 100 * ct / total if total else 0
        label = f'{func} ({os.path.basename(file)})'
        print(f"  {label:<43} {ct:>7.2f}s  {pct:>5.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--top',   type=int, default=25)
    args = parser.parse_args()

    stats_file = run_profile(args.steps)
    report(stats_file, args.steps, top_n=args.top)


if __name__ == '__main__':
    main()
