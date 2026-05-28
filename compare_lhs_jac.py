#!/usr/bin/env python
"""
Compare lhs_jac_banded (Path A) vs lhs_jac_tot + store_bandM (Path B).
Run from /home/kitzmann/Code/VULCAN/neoVULCAN.
"""
import os, sys

os.environ["OMP_NUM_THREADS"] = "1"

_base = os.path.dirname(os.path.abspath(__file__))
_src  = os.path.join(_base, 'src')
for _p in (_base, _src):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.chdir(_base)

# ── Override ini_mix BEFORE importing anything that reads vulcan_cfg ──────────
import vulcan_cfg
vulcan_cfg.ini_mix   = 'const_mix'
vulcan_cfg.const_mix = {'H2': 0.855, 'He': 0.144, 'H2O': 5e-4, 'PH3': 6e-7}

# Disable photo so we don't need stellar-flux tables at run time
vulcan_cfg.use_photo = False

import numpy as np

# ── Build chem_funs if not current ────────────────────────────────────────────
python_executable = sys.executable
os.system(python_executable + ' make_chem_funs.py  > /dev/null 2>&1')
os.system(python_executable + ' make_chemistry_jax.py > /dev/null 2>&1')

import store, build_atm
import chem_funs
from chem_funs import ni, nr
from rates import ReadRate
from ros2 import Ros2
from vulcan_cfg import nz

species = chem_funs.spec_list

# ── Replicate vulcan.py startup ───────────────────────────────────────────────
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

# ── Create solver and call naming_solver ──────────────────────────────────────
solver = Ros2()
solver.naming_solver(data_para)

# ── Path A: direct banded build ───────────────────────────────────────────────
lhs_b_A, bw  = solver.lhs_jac_banded(data_var, data_atm)

# ── Path B: dense → store_bandM ──────────────────────────────────────────────
lhs_dense    = solver.lhs_jac_tot(data_var, data_atm)
lhs_b_B, bw2 = solver.store_bandM(lhs_dense, ni, nz)

# ── Compare ───────────────────────────────────────────────────────────────────
diff     = np.abs(lhs_b_A - lhs_b_B)
max_abs  = float(np.max(diff))

# relative difference (avoid div-by-zero)
denom    = np.abs(lhs_b_B)
mask     = denom > 0
rel      = np.where(mask, diff / denom, 0.0)
max_rel  = float(np.max(rel))

idx_abs  = np.unravel_index(np.argmax(diff), diff.shape)
idx_rel  = np.unravel_index(np.argmax(rel),  rel.shape)

print(f"\n=== LHS Jacobian comparison ===")
print(f"bw (Path A) = {bw},  bw2 (Path B) = {bw2}")
print(f"lhs_b shape : {lhs_b_A.shape}")
print(f"Max absolute difference : {max_abs:.6e}  at banded index {idx_abs}")
print(f"Max relative difference : {max_rel:.6e}  at banded index {idx_rel}")
print(f"  (A value at max-rel) = {lhs_b_A[idx_rel]:.6e}")
print(f"  (B value at max-rel) = {lhs_b_B[idx_rel]:.6e}")
print("=================================\n")
