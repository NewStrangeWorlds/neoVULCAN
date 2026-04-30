"""
Regression test for neoVULCAN.

Runs both the original VULCAN (from ../vulcan/) and neoVULCAN for a fixed
number of integration steps using identical inputs, then compares the final
species number-density arrays (var.y) to within a tight relative tolerance.

Usage:
    python tests/test_regression.py

Run from the neoVULCAN/ directory.  The test creates isolated temporary
working trees for each run so that neither source tree is modified.
"""

import os
import sys
import shutil
import pickle
import subprocess
import tempfile
import textwrap
import numpy as np

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

# Steps to run (keep small so the test finishes quickly, ~30 s)
TEST_STEPS = 300


# ---------------------------------------------------------------------------
# Minimal test config — overrides the fields that control run length and
# suppress all interactive/plotting behaviour.  Everything else is taken
# from the source directory's vulcan_cfg.py.
# ---------------------------------------------------------------------------
TEST_CFG_OVERRIDES = textwrap.dedent(f"""\
    # ---- regression-test overrides (appended by test_regression.py) ----
    count_max       = {TEST_STEPS}
    runtime         = 1e30       # don't stop on time, only on step count
    use_live_plot   = False
    use_plot_end    = False
    use_plot_evo    = False
    use_save_movie  = False
    use_flux_movie  = False
    save_evolution  = False
    out_name        = 'regression_test_output.vul'
    # Use const_mix so FastChem (external C++ binary) is not required
    ini_mix         = 'const_mix'
    const_mix       = {{'H2': 0.855, 'He': 0.144, 'H2O': 5e-4, 'PH3': 6e-7}}
    use_ini_cold_trap = False
""")


def _run_vulcan(src_dir: str, tmp_dir: str) -> dict:
    """
    Copy *src_dir* into *tmp_dir*, append test overrides to vulcan_cfg.py,
    run vulcan.py -n (skip chem_funs regeneration), and return the
    unpickled output dict.
    """
    # Copy entire source tree into tmp dir
    shutil.copytree(src_dir, tmp_dir, dirs_exist_ok=True)

    # Append test overrides to the config
    cfg_path = os.path.join(tmp_dir, 'vulcan_cfg.py')
    with open(cfg_path, 'a') as f:
        f.write('\n')
        f.write(TEST_CFG_OVERRIDES)

    # Run vulcan.py — use -n to skip regenerating chem_funs.py
    result = subprocess.run(
        [PYTHON, 'vulcan.py', '-n'],
        cwd=tmp_dir,
        capture_output=True,
        text=True,
        timeout=600,
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
