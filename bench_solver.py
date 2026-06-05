#!/usr/bin/env python
"""Compare scipy.linalg.lapack.dgbtrf+dgbtrs vs scipy.sparse.linalg.splu on a
representative LHS matrix from a real Earth integration step.

Drives the standard neoVULCAN startup, captures one ``lhs_b`` after a couple
of steps so the integration is on a realistic state, then times each solver
N times.

Run from /home/kitzmann/Code/VULCAN/neoVULCAN.
"""
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"

_base = os.path.dirname(os.path.abspath(__file__))
_src  = os.path.join(_base, 'src')
for _p in (_base, _src):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_base)

from neovulcan_runtime import get_cfg
cfg = get_cfg()
cfg.plotting.use_live_plot = False
cfg.plotting.use_plot_end  = False
cfg.plotting.use_plot_evo  = False

import numpy as np
import scipy.sparse as sp
from scipy.linalg.lapack import dgbtrf, dgbtrs
from scipy.sparse.linalg import splu

os.system(sys.executable + ' make_chemistry_jax.py > /dev/null 2>&1')

import store, build_atm
import chemistry_jax as chem_funs
from chemistry_jax import ni
from rates import ReadRate
from ros2 import Ros2
nz = cfg.atmosphere.nz

species = chem_funs.spec_list


def setup_and_capture():
    """Build a real lhs_b by running the full startup sequence."""
    data_var  = store.Variables()
    data_atm  = store.AtmData()
    data_para = store.Parameters()

    make_atm = build_atm.Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if cfg.condensation.use_condense:
        make_atm.sp_sat(data_atm)

    rate = ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    if cfg.network.use_lowT_limit_rates:
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

    data_var.dt = 1e-3  # representative step size
    solver = Ros2()
    solver.naming_solver(data_para)

    # Pull lhs_b in LAPACK band storage (3*bw+1, N)
    lhs_b, bw = solver.lhs_jac_banded(data_var, data_atm)
    return lhs_b, bw


def banded_lapack_to_csc(lhs_b, bw):
    """Convert LAPACK band storage (3*bw+1, N) → scipy.sparse.csc_matrix.

    In LAPACK format, the band data lives in rows [bw, 3*bw] (top bw rows are
    dgbtrf workspace).  Element a[i, j] is at lhs_b[2*bw + i - j, j].
    """
    band = lhs_b[bw:]                       # (2*bw+1, N) compact band form
    N = band.shape[1]
    # Build COO arrays then convert.
    rows_l, cols_l, vals_l = [], [], []
    for offset in range(-bw, bw + 1):
        # row_in_band = bw - offset (compact form: ab[bw + i - j, j])
        row_in_band = bw - offset
        diag = band[row_in_band]
        # this diagonal connects (i, j) where j - i = offset
        if offset >= 0:
            i_idx = np.arange(0, N - offset)
            j_idx = i_idx + offset
        else:
            j_idx = np.arange(0, N + offset)
            i_idx = j_idx - offset
        # vals at column j of the band
        v = diag[j_idx]
        # drop structural zeros to keep CSC light
        nz_mask = v != 0
        rows_l.append(i_idx[nz_mask])
        cols_l.append(j_idx[nz_mask])
        vals_l.append(v[nz_mask])
    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    vals = np.concatenate(vals_l)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()
    return A


def bench_lapack(lhs_b, bw, df, rhs, n_iter):
    """Time n_iter (factor + 2× solve) using LAPACK banded LU."""
    times = []
    for _ in range(n_iter):
        ab = lhs_b.copy()                   # restore original (dgbtrf overwrites)
        t0 = time.perf_counter()
        ab_f, ipiv, info = dgbtrf(ab, bw, bw, overwrite_ab=1)
        k1, _ = dgbtrs(ab_f, bw, bw, df, ipiv)
        k2, _ = dgbtrs(ab_f, bw, bw, rhs, ipiv)
        times.append(time.perf_counter() - t0)
    return np.array(times)


def bench_splu(lhs_b, bw, df, rhs, n_iter):
    """Time n_iter (banded→CSC + splu + 2× solve)."""
    times = []
    times_factor = []
    times_solve = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        A = banded_lapack_to_csc(lhs_b, bw)
        lu = splu(A)
        t_fact = time.perf_counter() - t0
        k1 = lu.solve(df)
        k2 = lu.solve(rhs)
        times.append(time.perf_counter() - t0)
        times_factor.append(t_fact)
        times_solve.append(times[-1] - t_fact)
    return np.array(times), np.array(times_factor), np.array(times_solve)


def bench_splu_prebuilt(A, df, rhs, n_iter):
    """Time n_iter (splu + 2× solve), excluding the banded→CSC conversion."""
    times = []
    times_factor = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        lu = splu(A)
        t_fact = time.perf_counter() - t0
        k1 = lu.solve(df)
        k2 = lu.solve(rhs)
        times.append(time.perf_counter() - t0)
        times_factor.append(t_fact)
    return np.array(times), np.array(times_factor)


def main():
    print("Capturing representative lhs_b…")
    lhs_b, bw = setup_and_capture()
    N = lhs_b.shape[1]
    print(f"  lhs_b shape = {lhs_b.shape}, bw = {bw}, N = {N}")

    # Build a representative RHS pair (random but consistent).
    rng = np.random.default_rng(0)
    df  = rng.standard_normal(N)
    rhs = rng.standard_normal(N)

    # Pre-build CSC once to measure conversion cost separately.
    print("\nConverting banded → CSC for sparsity inspection…")
    t0 = time.perf_counter()
    A = banded_lapack_to_csc(lhs_b, bw)
    t_conv = time.perf_counter() - t0
    print(f"  CSC nnz = {A.nnz:,} / dense {N*N:,} ({100*A.nnz/(N*N):.2f}%)")
    band_entries = (2 * bw + 1) * N
    print(f"  band nnz fraction (CSC nnz / band entries) = "
          f"{100*A.nnz/band_entries:.2f}%")
    print(f"  banded→CSC conversion: {t_conv*1e3:.2f} ms")

    # Cross-check correctness once.
    print("\nCorrectness cross-check (one factor + solve via each path):")
    ab = lhs_b.copy()
    ab_f, ipiv, info = dgbtrf(ab, bw, bw, overwrite_ab=1)
    x_lap, _ = dgbtrs(ab_f, bw, bw, df, ipiv)
    lu = splu(A)
    x_splu = lu.solve(df)
    print(f"  max|x_lapack - x_splu|             = {float(np.max(np.abs(x_lap - x_splu))):.3e}")
    print(f"  max|x_lapack - x_splu| / max|x|    = "
          f"{float(np.max(np.abs(x_lap - x_splu)) / np.max(np.abs(x_lap))):.3e}")

    n_iter = 30
    # Warmup once each so first-call effects don't dominate.
    bench_lapack(lhs_b, bw, df, rhs, 2)
    bench_splu(lhs_b, bw, df, rhs, 2)
    bench_splu_prebuilt(A, df, rhs, 2)

    print(f"\nTiming each path × {n_iter} reps (median ms):")

    t_lap = bench_lapack(lhs_b, bw, df, rhs, n_iter)
    t_splu_total, t_splu_fact, t_splu_solve = bench_splu(lhs_b, bw, df, rhs, n_iter)
    t_splu_prebuilt_total, t_splu_prebuilt_fact = bench_splu_prebuilt(A, df, rhs, n_iter)

    def stats(arr):
        return float(np.median(arr) * 1e3), float(np.min(arr) * 1e3)

    print("\n  Operation                             median(ms)  min(ms)")
    print("  ----------------------------------------------------------")
    print("  LAPACK dgbtrf + 2*dgbtrs              {:>10.2f}  {:>7.2f}".format(*stats(t_lap)))
    print("  SuperLU (banded→CSC + splu + 2 solv)  {:>10.2f}  {:>7.2f}".format(*stats(t_splu_total)))
    print("    of which: banded→CSC + splu         {:>10.2f}  {:>7.2f}".format(*stats(t_splu_fact)))
    print("    of which: 2 solves                  {:>10.2f}  {:>7.2f}".format(*stats(t_splu_solve)))
    print("  SuperLU pre-built CSC (splu + 2 solv) {:>10.2f}  {:>7.2f}".format(*stats(t_splu_prebuilt_total)))
    print("    of which: splu only                 {:>10.2f}  {:>7.2f}".format(*stats(t_splu_prebuilt_fact)))


if __name__ == '__main__':
    main()
