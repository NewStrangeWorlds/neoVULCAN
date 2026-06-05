#!/usr/bin/env python
"""Compare diffdf between neoVULCAN and original VULCAN."""

import os, sys

# Working directory must be neoVULCAN so that vulcan_cfg, thermo/, atm/, etc. are found
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add neoVULCAN src/ to path
_src = os.path.join(script_dir, 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

# Patch vulcan_cfg before any imports that read it at module level
from neovulcan_runtime import get_cfg
cfg = get_cfg()
cfg.elements.ini_mix = 'const_mix'
cfg.elements.const_mix = {'H2': 0.855, 'He': 0.144, 'H2O': 5e-4, 'PH3': 6e-7}
cfg.photochemistry.use_photo = False  # skip photochemistry setup for this test

import numpy as np

# Build atmosphere (same steps as vulcan.py)
import store, build_atm

data_var = store.Variables()
data_atm = store.AtmData()
data_para = store.Parameters()

make_atm = build_atm.Atm()

data_atm = make_atm.f_pico(data_atm)
data_atm = make_atm.load_TPK(data_atm)

from rates import ReadRate
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
output_obj = Output()
data_atm = make_atm.f_mu_dz(data_var, data_atm, output_obj)
make_atm.mol_diff(data_atm)
make_atm.BC_flux(data_atm)

# ---- neoVULCAN diffdf ----
from ros2 import Ros2
solver_neo = Ros2()
result_neo = solver_neo.diffdf(data_var.y, data_atm).flatten()

# ---- original VULCAN diffdf ----
# Add original vulcan to path; its modules will shadow neoVULCAN's if not careful,
# but we only need op.Ros2 which is self-contained enough.
orig_vulcan = '/home/kitzmann/Code/VULCAN/vulcan'
if orig_vulcan not in sys.path:
    sys.path.insert(0, orig_vulcan)

import importlib, types

# We need to import op.py from the original vulcan WITHOUT disturbing the already-
# imported neoVULCAN modules.  Use importlib with a unique module name.
spec = importlib.util.spec_from_file_location(
    "orig_op", os.path.join(orig_vulcan, "op.py"))
orig_op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig_op)

orig_solver = orig_op.Ros2()
result_orig = orig_solver.diffdf(data_var.y, data_atm).flatten()

# ---- comparison ----
abs_diff = np.abs(result_neo - result_orig)
max_abs = np.max(abs_diff)

denom = np.abs(result_orig)
with np.errstate(divide='ignore', invalid='ignore'):
    rel_diff = np.where(denom > 0, abs_diff / denom, 0.0)
max_rel = np.max(rel_diff)

print(f"Max absolute difference: {max_abs:.6e}")
print(f"Max relative difference: {max_rel:.6e}")
