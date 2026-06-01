"""
Regression tests for neoVULCAN.

Two complementary test styles are provided:

1.  test_regression — STRUCTURAL gate.  Pins step count and demands
    bit-identical output (rtol=1e-8) between original VULCAN and neoVULCAN.
    Valid only for changes that preserve the step schedule (refactors,
    dead-code removal, JAX chemistry vs auto-generated chemistry, etc.).
    Solver-algorithm changes (PI controller, higher-order Rosenbrock, log
    space, Newton finisher) *will* break this test by design — use the
    fixed-time tests below for those.

2.  test_regression_short_time / test_regression_long_time — NUMERICAL gate.
    Integrate both codes to the same physical time and compare with a
    tolerance comparable to the solver's per-step rtol.  These are the
    correct gates for any change that alters the numerical algorithm.

Usage:
    python -m pytest tests/test_regression.py -v
    python -m pytest tests/test_regression.py -v -m "not slow"  # skip the long one

Run from the neoVULCAN/ directory.  Each test creates isolated temporary
working trees so neither source tree is modified.
"""

import os
import sys
import shutil
import pickle
import subprocess
import tempfile
import textwrap
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths (relative to neoVULCAN/)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
NEO_DIR = os.path.dirname(HERE)                          # neoVULCAN/
ORIG_DIR = os.path.join(os.path.dirname(NEO_DIR), 'vulcan')   # ../vulcan/

PYTHON = sys.executable

# Tolerance for the species array comparison
RTOL = 1e-8
ATOL = 1e-30   # absolute floor for near-zero values

# Steps to run for the strict structural test (keep small ~30 s)
TEST_STEPS = 300


# ---------------------------------------------------------------------------
# Common config overrides shared by every test.  Suppresses interactive
# behaviour and pins the chemistry input.
# ---------------------------------------------------------------------------
_COMMON_OVERRIDES = textwrap.dedent("""\
    # ---- regression-test common overrides ----
    use_photo       = False      # disable photochemistry so both codes use identical physics
    use_live_plot   = False
    use_plot_end    = False
    use_plot_evo    = False
    use_save_movie  = False
    use_flux_movie  = False
    save_evolution  = False
    out_name        = 'regression_test_output.vul'
    # Use const_mix so FastChem (external C++ binary) is not required
    ini_mix         = 'const_mix'
    const_mix       = {'H2': 0.855, 'He': 0.144, 'H2O': 5e-4, 'PH3': 6e-7}
    use_ini_cold_trap = False
""")

# Strict structural test: stops on step count, not time.
TEST_CFG_OVERRIDES = _COMMON_OVERRIDES + textwrap.dedent(f"""\
    count_max       = {TEST_STEPS}
    runtime         = 1e30
""")


def _physical_time_overrides(runtime_sec: float) -> str:
    """Build cfg overrides that stop on physical time, not step count.

    Pins solver rtol to a tight value (0.02 = rtol_min) so the two integrators
    are forced to converge tightly enough that algorithmic differences (step
    controller, integrator order, etc.) don't dominate the comparison.
    """
    return _COMMON_OVERRIDES + textwrap.dedent(f"""\
        runtime         = {runtime_sec}
        count_max       = 1000000   # effectively unbounded
        # Tight solver rtol so two correct integrators must agree closely.
        rtol            = 0.02
        rtol_min        = 0.02
        rtol_max        = 0.02
        use_adapt_rtol  = False
        # Tight steady-state criterion so neither run short-circuits early.
        yconv_cri       = 1e-30
        slope_cri       = 1e-30
    """)


def _run_vulcan(src_dir: str, tmp_dir: str, cfg_overrides: str = TEST_CFG_OVERRIDES) -> dict:
    """
    Copy *src_dir* into *tmp_dir*, append *cfg_overrides* to vulcan_cfg.py,
    run vulcan.py -n (skip chem_funs regeneration), and return the
    unpickled output dict.
    """
    # Copy entire source tree into tmp dir
    shutil.copytree(src_dir, tmp_dir, dirs_exist_ok=True)

    # Append test overrides to the config
    cfg_path = os.path.join(tmp_dir, 'vulcan_cfg.py')
    with open(cfg_path, 'a') as f:
        f.write('\n')
        f.write(cfg_overrides)

    # Run vulcan.py — use -n to skip regenerating chem_funs.py
    result = subprocess.run(
        [PYTHON, 'vulcan.py', '-n'],
        cwd=tmp_dir,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode != 0:
        print('--- STDOUT ---')
        print(result.stdout[-3000:])
        print('--- STDERR ---')
        print(result.stderr[-3000:])
        raise RuntimeError(f'vulcan.py exited with code {result.returncode}')

    output_file = os.path.join(tmp_dir, 'output', 'regression_test_output.vul')
    if not os.path.exists(output_file):
        raise FileNotFoundError(f'Output file not found: {output_file}')

    with open(output_file, 'rb') as f:
        return pickle.load(f)


@pytest.mark.skip(
    reason="Strict bit-identical test against original VULCAN.  Diverges once "
           "neoVULCAN's numerical algorithm differs from VULCAN's "
           "(PI controller, higher-order Rosenbrock, etc.).  Kept as a "
           "diagnostic tool — un-skip locally to validate a structural-only "
           "change (e.g. a pure refactor) that should be bit-identical."
)
def test_regression():
    with tempfile.TemporaryDirectory(prefix='vulcan_orig_') as orig_tmp, \
         tempfile.TemporaryDirectory(prefix='vulcan_neo_')  as neo_tmp:

        print(f'Running original VULCAN from {ORIG_DIR} ...')
        orig_out = _run_vulcan(ORIG_DIR, orig_tmp)

        print(f'Running neoVULCAN from {NEO_DIR} ...')
        neo_out  = _run_vulcan(NEO_DIR,  neo_tmp)

    orig_y = orig_out['variable']['y']   # shape (nz, ni)
    neo_y  = neo_out['variable']['y']

    if orig_y.shape != neo_y.shape:
        raise AssertionError(
            f'Shape mismatch: original {orig_y.shape} vs neoVULCAN {neo_y.shape}'
        )

    # Compare: allow a small absolute floor so near-zero species don't dominate
    close = np.allclose(orig_y, neo_y, rtol=RTOL, atol=ATOL)

    if not close:
        diff = np.abs(orig_y - neo_y)
        rel  = diff / (np.abs(orig_y) + ATOL)
        worst_idx = np.unravel_index(np.argmax(rel), rel.shape)
        worst_rel = rel[worst_idx]
        print(f'Max relative difference: {worst_rel:.3e} at layer/species {worst_idx}')
        print(f'  original value : {orig_y[worst_idx]:.6e}')
        print(f'  neoVULCAN value: {neo_y[worst_idx]:.6e}')
        raise AssertionError(
            f'Species arrays differ by up to {worst_rel:.2e} relative error '
            f'(tolerance {RTOL:.0e})'
        )

    print(f'PASSED — y arrays match within rtol={RTOL:.0e} over {TEST_STEPS} steps.')
    print(f'  Shape: {orig_y.shape}  |  max abs diff: {np.max(np.abs(orig_y - neo_y)):.2e}')


# ---------------------------------------------------------------------------
# Numerical gate: integrate both codes to a fixed physical time and compare.
# Use this for any change that alters the numerical algorithm (step
# controller, integrator order, log-space, splitting, Newton finisher).
# Tolerance is comparable to the solver per-step rtol (0.25) — bit-identicality
# is NOT expected because step schedules differ.
# ---------------------------------------------------------------------------

def _compare_fixed_time(runtime_sec: float, rtol: float, label: str,
                        neo_overrides_extra: str = ''):
    """Compare original VULCAN vs neoVULCAN at a fixed physical time.

    ``neo_overrides_extra`` lets the neoVULCAN run override config
    settings the original doesn't support (e.g. ``ode_solver='Rodas3'``).
    The original VULCAN run always uses its default solver (Ros2).
    """
    overrides_orig = _physical_time_overrides(runtime_sec)
    overrides_neo  = overrides_orig + neo_overrides_extra
    with tempfile.TemporaryDirectory(prefix='vulcan_orig_') as orig_tmp, \
         tempfile.TemporaryDirectory(prefix='vulcan_neo_')  as neo_tmp:

        print(f'[{label}] Running original VULCAN to t={runtime_sec:.0e} s ...')
        orig_out = _run_vulcan(ORIG_DIR, orig_tmp, overrides_orig)
        print(f'[{label}] Running neoVULCAN  to t={runtime_sec:.0e} s ...')
        neo_out  = _run_vulcan(NEO_DIR,  neo_tmp, overrides_neo)

    orig_y = orig_out['variable']['y']
    neo_y  = neo_out['variable']['y']
    orig_t = orig_out['variable']['t']
    neo_t  = neo_out['variable']['t']

    if orig_y.shape != neo_y.shape:
        raise AssertionError(
            f'Shape mismatch: original {orig_y.shape} vs neoVULCAN {neo_y.shape}')

    # Both runs should land near the same physical time (within one step of the target).
    print(f'[{label}] original final t = {orig_t:.3e} s,  neo final t = {neo_t:.3e} s')

    # Compare on the per-species mixing ratio (sum-normalised) to remove a
    # spurious sensitivity to the total number density at the run endpoint.
    orig_x = orig_y / orig_y.sum(axis=1, keepdims=True)
    neo_x  = neo_y  / neo_y.sum(axis=1, keepdims=True)

    abs_floor = 1e-25
    diff = np.abs(orig_x - neo_x)
    rel  = diff / (np.abs(orig_x) + abs_floor)
    # Ignore species whose mixing ratio is below 1e-8 in BOTH runs.  Trace
    # species below this threshold are physically negligible and numerically
    # dominated by the per-step rtol; comparing them produces noise, not signal.
    mixing_ratio_threshold = 1e-8
    mask = (np.abs(orig_x) > mixing_ratio_threshold) | (np.abs(neo_x) > mixing_ratio_threshold)
    rel_masked = np.where(mask, rel, 0.0)
    worst_rel = float(rel_masked.max())
    worst_idx = np.unravel_index(np.argmax(rel_masked), rel.shape)

    print(f'[{label}] worst relative mixing-ratio diff: {worst_rel:.3e} '
          f'at (layer, species) = {worst_idx}')

    if worst_rel > rtol:
        print(f'  original x : {orig_x[worst_idx]:.6e}')
        print(f'  neoVULCAN x: {neo_x[worst_idx]:.6e}')
        raise AssertionError(
            f'[{label}] mixing ratios differ by up to {worst_rel:.2e} '
            f'(tolerance {rtol:.0e})')

    print(f'[{label}] PASSED — max relative mixing-ratio diff {worst_rel:.2e} '
          f'within rtol={rtol:.0e}')


def test_regression_short_time():
    """Cheap solver-change gate: t=1e4 s, rtol=5e-2."""
    _compare_fixed_time(runtime_sec=1e4, rtol=5e-2, label='short')


@pytest.mark.slow
def test_regression_long_time():
    """Deeper solver-change gate: t=1e6 s, rtol=1e-1.

    Closer to steady state.  Slow (~few minutes).  Skip with `-m "not slow"`.
    """
    _compare_fixed_time(runtime_sec=1e6, rtol=1e-1, label='long')


def test_regression_rodas3_short_time():
    """Rodas3 vs original VULCAN: t=1e4 s, rtol=1e-1.

    Cross-integrator comparison.  Both runs use the same per-step
    solver rtol=0.02 (from _physical_time_overrides) but converge via
    different algorithms — empirically diverge by ~5-10% on mixing
    ratios at t=1e4 s, so we accept up to 10% (5× the per-step rtol).
    A bug would manifest as orders-of-magnitude divergence or NaNs,
    not as a few-percent drift.
    """
    _compare_fixed_time(runtime_sec=1e4, rtol=1e-1, label='rodas3-short',
                        neo_overrides_extra="ode_solver = 'Rodas3'\n")


if __name__ == '__main__':
    # Also check that ORIG_DIR exists before attempting anything
    if not os.path.isdir(ORIG_DIR):
        print(f'ERROR: original VULCAN directory not found at {ORIG_DIR}')
        print('Expected layout:  VULCAN/vulcan/  (original)  and  VULCAN/neoVULCAN/  (fork)')
        sys.exit(1)

    try:
        test_regression()
    except (AssertionError, RuntimeError, FileNotFoundError) as e:
        print(f'FAILED: {e}')
        sys.exit(1)
