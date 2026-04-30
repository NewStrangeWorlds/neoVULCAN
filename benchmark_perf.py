#!/usr/bin/env python3
"""
benchmark_perf.py — wall-time comparison of old VULCAN vs neoVULCAN (JAX).

Benchmarks the two hot-path functions:
  chemdf       — called twice per Ros2 step (chemistry RHS)
  neg_achemjac — called once  per Ros2 step (chemistry Jacobian)

The old version uses hand-coded NumPy lambda expressions (chem_funs.py).
The new version uses JAX-native expressions + jax.jacfwd (chemistry_jax.py).

Run from neoVULCAN/:
    python benchmark_perf.py
"""

import os
import sys
import subprocess
import numpy as np
import tempfile
import pickle

HERE   = os.path.dirname(os.path.abspath(__file__))
OLD_DIR = os.path.join(os.path.dirname(HERE), 'vulcan')

# Network dimensions for PHO_full_photo_network
NZ = 120   # atmospheric layers
NI = 32    # species
NR = 424   # reactions (forward + reverse)

N_WARMUP = 3    # calls to discard (JAX JIT compilation, cache warm-up)
N_TIMED  = 50   # calls to time

SEED = 42


def make_inputs():
    """Generate reproducible synthetic inputs matching PHO network dimensions."""
    rng = np.random.default_rng(SEED)
    y      = np.abs(rng.standard_normal((NZ, NI))) * 1e8 + 1e2   # (nz, ni), positive
    M      = np.abs(rng.standard_normal(NZ)) * 1e16 + 1e15       # (nz,),   positive
    k_dict = {i: np.abs(rng.standard_normal(NZ)) * 1e-15 + 1e-18
              for i in range(1, NR + 1)}
    return y, M, k_dict


# ---------------------------------------------------------------------------
# Old VULCAN timing script (runs via subprocess in vulcan/ dir)
# ---------------------------------------------------------------------------

_OLD_SCRIPT = r"""
import sys, os, pickle, time
import numpy as np

old_dir = sys.argv[1]
inp_file = sys.argv[2]
n_warmup = int(sys.argv[3])
n_timed  = int(sys.argv[4])

sys.path.insert(0, old_dir)

# Load the old chem_funs (which imports vulcan_cfg from old_dir)
import chem_funs
chemdf_old    = chem_funs.chemdf
neg_symjac_old = chem_funs.neg_symjac

with open(inp_file, 'rb') as f:
    y, M, k_dict = pickle.load(f)

# Warm up
for _ in range(n_warmup):
    chemdf_old(y, M, k_dict)
    neg_symjac_old(y, M, k_dict)

# Time chemdf
t0 = time.perf_counter()
for _ in range(n_timed):
    chemdf_old(y, M, k_dict)
t1 = time.perf_counter()
chemdf_ms = (t1 - t0) / n_timed * 1e3

# Time neg_symjac
t0 = time.perf_counter()
for _ in range(n_timed):
    neg_symjac_old(y, M, k_dict)
t1 = time.perf_counter()
jac_ms = (t1 - t0) / n_timed * 1e3

print(f"chemdf_ms={chemdf_ms:.4f}")
print(f"jac_ms={jac_ms:.4f}")
"""

# ---------------------------------------------------------------------------
# New neoVULCAN timing script (runs via subprocess in neoVULCAN/ dir)
# ---------------------------------------------------------------------------

_NEW_SCRIPT = r"""
import sys, os, pickle, time
import numpy as np

neo_dir = sys.argv[1]
inp_file = sys.argv[2]
n_warmup = int(sys.argv[3])
n_timed  = int(sys.argv[4])

sys.path.insert(0, os.path.join(neo_dir, 'src'))

from chemistry_jax import chemdf, neg_achemjac

with open(inp_file, 'rb') as f:
    y, M, k_dict = pickle.load(f)

# Warm up (triggers JAX JIT compilation)
for _ in range(n_warmup):
    chemdf(y, M, k_dict)
    neg_achemjac(y, M, k_dict)

# Time chemdf
t0 = time.perf_counter()
for _ in range(n_timed):
    chemdf(y, M, k_dict)
t1 = time.perf_counter()
chemdf_ms = (t1 - t0) / n_timed * 1e3

# Time neg_achemjac
t0 = time.perf_counter()
for _ in range(n_timed):
    neg_achemjac(y, M, k_dict)
t1 = time.perf_counter()
jac_ms = (t1 - t0) / n_timed * 1e3

print(f"chemdf_ms={chemdf_ms:.4f}")
print(f"jac_ms={jac_ms:.4f}")
"""


def _parse_output(stdout: str) -> dict:
    result = {}
    for line in stdout.splitlines():
        line = line.strip()
        if '=' in line and '_ms=' in line:
            k, v = line.split('=')
            result[k] = float(v)
    return result


def run_subprocess_timing(script_src: str, *args) -> dict:
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
        f.write(script_src)
        script_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, script_path, *[str(a) for a in args]],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print('STDERR:', result.stderr[-2000:])
            raise RuntimeError(f'Subprocess exited with code {result.returncode}')
        return _parse_output(result.stdout)
    finally:
        os.unlink(script_path)


def main():
    print(f"VULCAN vs neoVULCAN performance comparison")
    print(f"Network: PHO ({NZ} layers, {NI} species, {NR} reactions)")
    print(f"Warmup calls: {N_WARMUP}   Timed calls: {N_TIMED}")
    print()

    # Shared inputs
    y, M, k_dict = make_inputs()
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pickle.dump((y, M, k_dict), f)
        inp_file = f.name

    try:
        print("Timing old VULCAN (chem_funs.py lambda expressions)...")
        old = run_subprocess_timing(_OLD_SCRIPT, OLD_DIR, inp_file, N_WARMUP, N_TIMED)

        print("Timing neoVULCAN (chemistry_jax.py, JAX JIT)...")
        new = run_subprocess_timing(_NEW_SCRIPT, HERE, inp_file, N_WARMUP, N_TIMED)

    finally:
        os.unlink(inp_file)

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print()
    print(f"{'Function':<22}  {'old VULCAN':>12}  {'neoVULCAN':>12}  {'speedup':>10}")
    print("-" * 62)

    for fn_label, old_key, new_key in [
        ('chemdf',        'chemdf_ms', 'chemdf_ms'),
        ('neg_*chemjac',  'jac_ms',    'jac_ms'),
    ]:
        o = old.get(old_key, float('nan'))
        n = new.get(new_key, float('nan'))
        speedup = o / n if n > 0 else float('nan')
        direction = "faster" if speedup > 1 else "slower"
        print(f"{fn_label:<22}  {o:>10.3f}ms  {n:>10.3f}ms  {speedup:>7.2f}x {direction}")

    print()
    o_total = old.get('chemdf_ms', 0) * 2 + old.get('jac_ms', 0)
    n_total = new.get('chemdf_ms', 0) * 2 + new.get('jac_ms', 0)
    print(f"Estimated per-step total (2×chemdf + 1×jac):")
    print(f"  old VULCAN : {o_total:.3f} ms")
    print(f"  neoVULCAN  : {n_total:.3f} ms")
    if n_total > 0:
        print(f"  speedup    : {o_total/n_total:.2f}x")


if __name__ == '__main__':
    main()
