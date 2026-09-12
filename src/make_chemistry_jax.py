#!/usr/bin/env python3
"""
make_chemistry_jax.py — generates src/chemistry_jax.py from a VULCAN network file.

Usage:
    python make_chemistry_jax.py [-c vulcan_cfg.toml]

Reads the network specified in cfg.network.network and writes
src/chemistry_jax.py, which provides the chemistry right-hand side
(``chemdf``), its Jacobian (``chem_jac_blocks``) and the equilibrium constants
(``Gibbs``) for that network.

Run this whenever the chemical network changes, before running vulcan.py
(vulcan.py does it automatically unless started with ``-n``).

Design
------
The generated module is *table driven*: the network is compiled into a few
small integer/float arrays (reactant indices and stoichiometries per reaction
direction, third-body powers, and pre-sorted scatter lists for the RHS and
the Jacobian), and the kernels that consume them are fixed code.  Every
reaction is mass action, ``rate_d = k_d * M**m_d * prod_r y_r**s_r``, so this
covers the whole network, photolysis and reverse directions included (their
rate coefficients simply arrive through ``k``).

The previous generator wrote one explicit expression per species / per
Jacobian entry (~20 000 lines for the SNCHO network).  That traced through
JAX in ~13 s on every start-up; the table form traces in milliseconds and
evaluates at the same speed.
"""

import argparse
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))            # .../neoVULCAN/src
_root = os.path.dirname(_here)                                # .../neoVULCAN
sys.path.insert(0, _here)
sys.path.insert(0, _root)

_OFNAME = os.path.join(_here, 'chemistry_jax.py')


# ---------------------------------------------------------------------------
# Network parser
# ---------------------------------------------------------------------------

def parse_network(path):
    """Parse a VULCAN network file and return (chem_dict, reactions).

    chem_dict : {species_name: index}  in first-appearance order
    reactions : list of (j, reac, prod, rxn_str)
        j        forward reaction index, sequential 1, 3, 5, ...
        reac     [[stoi, name], ...]  all reactants, including M if present
        prod     [[stoi, name], ...]  all products,  including M if present
        rxn_str  raw "A + B -> C + D" string for comments
    """
    chem_dict = {}
    reactions = []
    j = -1

    re_end = False

    with open(path) as f:
        for raw in f:
            line = raw.strip()

            if line.startswith('# re_end'):
                re_end = True
                continue
            if re_end:
                continue
            if line.startswith('#') or not line:
                continue

            inner = raw.partition('[')[-1].rpartition(']')[0].strip()
            if not inner:
                continue

            j += 2

            reac, prod, reading_reactants = [], [], True
            for token in inner.split():
                if token == '+':
                    continue
                if token == '->':
                    reading_reactants = False
                    continue

                parts = token.split('*')
                if len(parts) == 1:
                    stoi, name = 1, parts[0]
                else:
                    stoi, name = int(parts[0]), parts[1]

                if name != 'M' and name not in chem_dict:
                    chem_dict[name] = len(chem_dict)

                if reading_reactants:
                    reac.append([stoi, name])
                else:
                    prod.append([stoi, name])

            reactions.append((j, reac, prod, inner))

    return chem_dict, reactions


def _merge_species(species_noM, chem_dict):
    """Sum stoichiometries of species listed more than once (OH + OH -> 2 OH).

    Returns [(idx, stoi), ...] sorted by species index."""
    merged = {}
    for stoi, name in species_noM:
        idx = chem_dict[name]
        merged[idx] = merged.get(idx, 0) + stoi
    return sorted(merged.items())


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def build_tables(chem_dict, reactions):
    """Compile the network into the arrays the kernels consume.

    Directions are numbered like VULCAN's rate dict: forward ``j`` (odd) and
    reverse ``j+1`` (even), 1-based; row 0 of every per-direction table is an
    unused pad.  A padded reactant slot points at the extra species index
    ``ni`` (whose number density is defined as 1) with stoichiometry 0.
    """
    ni = len(chem_dict)
    nr = 2 * len(reactions)

    # per direction: reactant slots [(idx, stoi)], M power, net {idx: coef}
    dirs = {}
    for j, reac, prod, _rxn in reactions:
        reac_noM = [(s, n) for s, n in reac if n != 'M']
        prod_noM = [(s, n) for s, n in prod if n != 'M']
        m_f = sum(s for s, n in reac if n == 'M')
        m_r = sum(s for s, n in prod if n == 'M')
        rm = _merge_species(reac_noM, chem_dict)
        pm = _merge_species(prod_noM, chem_dict)
        net = {}
        for idx, s in rm:
            net[idx] = net.get(idx, 0) - s
        for idx, s in pm:
            net[idx] = net.get(idx, 0) + s
        net = {i: c for i, c in sorted(net.items()) if c != 0}
        dirs[j] = (rm, m_f, net)
        dirs[j + 1] = (pm, m_r, {i: -c for i, c in net.items()})

    max_slots = max(len(rm) for rm, _, _ in dirs.values())
    max_stoi = max(s for rm, _, _ in dirs.values() for _, s in rm)
    max_mpow = max(m for _, m, _ in dirs.values())

    r_idx = np.full((nr + 1, max_slots), ni, dtype=np.int32)
    r_sto = np.zeros((nr + 1, max_slots), dtype=np.int32)
    m_pow = np.zeros(nr + 1, dtype=np.int32)
    rhs_c = []    # (species, pair, coef)   pair p <-> forward direction 2p+1
    jac_c = []    # (row species, col species, direction, slot, coef)
    for d in range(1, nr + 1):
        rm, m, net = dirs[d]
        m_pow[d] = m
        for slot, (idx, s) in enumerate(rm):
            r_idx[d, slot] = idx
            r_sto[d, slot] = s
        for i, c in net.items():
            if d % 2 == 1:                       # net rate of the pair, v = fwd - rev
                rhs_c.append((i, (d - 1) // 2, float(c)))
            for slot, (r, _s) in enumerate(rm):
                jac_c.append((i, r, d, slot, float(c)))

    # RHS scatter: sorted by receiving species, segment = species.  Each
    # contribution is the NET rate of a reversible pair (forward minus
    # reverse, cancelled before accumulation, as the explicit expressions
    # did) times the forward net stoichiometry.
    rhs_c.sort()
    rhs_sp = sorted({i for i, _, _ in rhs_c})
    seg_of = {i: n for n, i in enumerate(rhs_sp)}
    rhs = {
        'pair': np.array([p for _, p, _ in rhs_c], dtype=np.int32),
        'coef': np.array([c for _, _, c in rhs_c], dtype=np.float64),
        'seg':  np.array([seg_of[i] for i, _, _ in rhs_c], dtype=np.int32),
        'sp':   np.array(rhs_sp, dtype=np.int32),
    }

    # Jacobian scatter: sorted by (row, col), segment = non-zero (row, col) pair
    jac_c.sort()
    pairs = sorted({(i, r) for i, r, _, _, _ in jac_c})
    seg_of = {p: n for n, p in enumerate(pairs)}
    jac = {
        'd':    np.array([d for _, _, d, _, _ in jac_c], dtype=np.int32),
        'slot': np.array([s for _, _, _, s, _ in jac_c], dtype=np.int32),
        'coef': np.array([c for _, _, _, _, c in jac_c], dtype=np.float64),
        'seg':  np.array([seg_of[(i, r)] for i, r, _, _, _ in jac_c], dtype=np.int32),
        'rows': np.array([i for i, _ in pairs], dtype=np.int32),
        'cols': np.array([r for _, r in pairs], dtype=np.int32),
    }

    return dict(ni=ni, nr=nr, max_slots=max_slots, max_stoi=max_stoi,
                max_mpow=max_mpow, r_idx=r_idx, r_sto=r_sto, m_pow=m_pow,
                rhs=rhs, jac=jac)


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------

def _emit_array(name, arr, per_line=24):
    """Python source for a NumPy array literal (flat data + reshape)."""
    flat = arr.reshape(-1)
    if arr.dtype.kind == 'f':
        items = [repr(float(v)) for v in flat]
        dtype = 'np.float64'
    else:
        items = [str(int(v)) for v in flat]
        dtype = 'np.int32'
    lines = [f'{name} = np.array([']
    for start in range(0, len(items), per_line):
        lines.append('    ' + ', '.join(items[start:start + per_line]) + ',')
    lines.append(f'], dtype={dtype})')
    if arr.ndim > 1:
        lines[-1] += f'.reshape({arr.shape})'
    return '\n'.join(lines) + '\n'


def _emit_int_pow(name, smax):
    """``x**s`` for integer ``s`` in 0..smax without calling ``power``.

    Emitted as a where-chain so the exponent can be a traced array while the
    arithmetic stays plain multiplications (exact for s = 0 and s = 1)."""
    # innermost: s == 0 -> 1; outer: s == smax
    terms = {0: '1.0', 1: 'x'}
    for s in range(2, smax + 1):
        terms[s] = '(' + ' * '.join(['x'] * s) + ')'
    body = terms[0]
    for s in range(1, smax + 1):
        body = f'jnp.where(s == {s}, {terms[s]}, {body})'
    return (f'def {name}(x, s):\n'
            f'    """x**s for integer s in 0..{smax} (s may be a traced array)."""\n'
            f'    return {body}\n')


_HEADER = '''\
"""Chemistry functions for neoVULCAN.

AUTO-GENERATED by make_chemistry_jax.py — do not edit by hand.
Network : {network}
Species : {ni}    Reactions (fwd+rev): {nr}

Public API:
    chemdf(y, M, k_dict)          -> (nz, ni)       dn/dt from chemistry
    chem_jac_blocks(y, M, k_dict) -> (nz, ni, ni)   positive Jacobian d(dy_i/dt)/dy_r
    neg_achemjac(y, M, k_dict)    -> (ni*nz, ni*nz) negative block-diag Jacobian
    Gibbs(i, T)                   -> scalar          K_eq for forward reaction i
    k_dict_to_array(k_dict)       -> (nr+1, nz)     rate dict -> array
    spec_list                     : list[str]        species names in index order
    ni, nr                        : int              species count, reaction count

Single-layer JAX kernels (used by jacobian_jax.py):
    _chemdf_single(y, M, k), _jac_single(y, M, k) and their vmaps.

Layout
------
Every reaction direction d = 1..nr (forward j odd, reverse j+1 even, as in
VULCAN's rate dict; row 0 unused) is mass action,

    rate_d = k[d] * M**_M_POW[d] * prod_slot y[_R_IDX[d, slot]]**_R_STO[d, slot]

with padded reactant slots pointing at an extra species (index ni, y = 1,
stoichiometry 0).  The RHS and the Jacobian are pre-sorted scatter lists over
these rates:

    v[p]              = rate[2p+1] - rate[2p+2]          (net rate of pair p)
    dy[_RHS_SP[seg]]  = sum over contributions c with _RHS_SEG[c] == seg of
                        v[_RHS_PAIR[c]] * _RHS_COEF[c]
    J[_JAC_ROWS[seg], _JAC_COLS[seg]]
                      = sum over contributions c with _JAC_SEG[c] == seg of
                        d rate[_JAC_D[c]] / d y[slot _JAC_SLOT[c]] * _JAC_COEF[c]

where _RHS_COEF (forward direction) / _JAC_COEF are the signed net
stoichiometries.  Cancelling forward against reverse before accumulating
keeps the round-off of near-equilibrium pairs at the level of the net rate.
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import block_diag as _scipy_block_diag
from phy_const import kb, Navo

# Enable 64-bit floats (JAX defaults to float32; VULCAN uses float64 throughout).
jax.config.update("jax_enable_x64", True)
# Force JAX onto CPU for single-run workloads.
jax.config.update("jax_default_device", jax.devices("cpu")[0])


# ---------------------------------------------------------------------------
# Network metadata
# ---------------------------------------------------------------------------

spec_list = {spec_list!r}
ni = {ni}
nr = {nr}

# Reactant slots per direction, largest stoichiometry, largest M power.
_MAX_SLOTS = {max_slots}
_MAX_STOI  = {max_stoi}
_MAX_MPOW  = {max_mpow}


# ---------------------------------------------------------------------------
# Network tables (NumPy; JAX copies below)
# ---------------------------------------------------------------------------

'''

_KERNELS = '''

_R_IDX    = jnp.asarray(_R_IDX_NP)
_R_STO    = jnp.asarray(_R_STO_NP)
_M_POW    = jnp.asarray(_M_POW_NP)
_RHS_PAIR = jnp.asarray(_RHS_PAIR_NP)
_RHS_COEF = jnp.asarray(_RHS_COEF_NP)
_RHS_SEG  = jnp.asarray(_RHS_SEG_NP)
_RHS_SP   = jnp.asarray(_RHS_SP_NP)
_RHS_NSEG = int(_RHS_SP_NP.shape[0])
_JAC_D    = jnp.asarray(_JAC_D_NP)
_JAC_SLOT = jnp.asarray(_JAC_SLOT_NP)
_JAC_COEF = jnp.asarray(_JAC_COEF_NP)
_JAC_SEG  = jnp.asarray(_JAC_SEG_NP)
_JAC_ROWS = jnp.asarray(_JAC_ROWS_NP)
_JAC_COLS = jnp.asarray(_JAC_COLS_NP)
_JAC_NSEG = int(_JAC_ROWS_NP.shape[0])
_SLOTS    = jnp.arange(_MAX_SLOTS)


# ---------------------------------------------------------------------------
# JAX single-layer kernels:  y(ni,), M(scalar), k(nr+1,)
# ---------------------------------------------------------------------------

{int_pow}

def _rate_factors(y, M, k):
    """Per-direction pieces of the mass-action rate.

    Returns (kM, yr, f): ``kM = k * M**m`` (nr+1,), the reactant number
    densities per slot ``yr`` (nr+1, slots) and their stoichiometric powers
    ``f = yr**s`` (padded slots give 1).
    """
    yp = jnp.concatenate([y, jnp.ones((1,), dtype=y.dtype)])
    yr = yp[_R_IDX]
    f  = _int_pow(yr, _R_STO)
    kM = k * _int_pow(M, _M_POW)
    return kM, yr, f


def _chemdf_single(y, M, k):
    """dn/dt from chemistry for one layer: (ni,)."""
    kM, _yr, f = _rate_factors(y, M, k)
    rate = kM * jnp.prod(f, axis=1)
    v = rate[1::2] - rate[2::2]                      # net rate per reversible pair
    vals = jax.ops.segment_sum(v[_RHS_PAIR] * _RHS_COEF, _RHS_SEG,
                               num_segments=_RHS_NSEG, indices_are_sorted=True)
    return jnp.zeros((ni,), dtype=y.dtype).at[_RHS_SP].set(vals)


def _jac_single(y, M, k):
    """Analytical chemistry Jacobian J[i, r] = d(dy_i/dt)/d(y_r) for one layer: (ni, ni)."""
    kM, yr, f = _rate_factors(y, M, k)
    # d rate_d / d y_slot = kM * s * y_slot**(s-1) * prod_{other slots} f
    excl = jnp.stack([jnp.prod(jnp.where(_SLOTS == j, 1.0, f), axis=1)
                      for j in range(_MAX_SLOTS)], axis=1)
    dpow = _R_STO * _int_pow(yr, jnp.maximum(_R_STO - 1, 0))
    dr = kM[:, None] * dpow * excl
    vals = jax.ops.segment_sum(dr[_JAC_D, _JAC_SLOT] * _JAC_COEF, _JAC_SEG,
                               num_segments=_JAC_NSEG, indices_are_sorted=True)
    return jnp.zeros((ni, ni), dtype=y.dtype).at[_JAC_ROWS, _JAC_COLS].set(vals)


_chemdf_vmap = jax.vmap(_chemdf_single, in_axes=(0, 0, 1))
chemdf_jax   = jax.jit(_chemdf_vmap)
_jac_vmap    = jax.vmap(_jac_single, in_axes=(0, 0, 1))
_jac_jit     = jax.jit(_jac_vmap)


# ---------------------------------------------------------------------------
# NumPy versions of the same tables (fallback, USE_JAX_CHEM = False)
# ---------------------------------------------------------------------------

def _rate_factors_numpy(y, M, k_arr):
    nz = y.shape[0]
    yp = np.concatenate([y, np.ones((nz, 1))], axis=1)
    yr = yp[:, _R_IDX_NP]                                    # (nz, nr+1, slots)
    f  = yr ** _R_STO_NP
    kM = k_arr.T * M[:, None] ** _M_POW_NP                   # (nz, nr+1)
    return kM, yr, f


def chemdf_numpy(y, M, k_arr):
    """y: (nz, ni), M: (nz,), k_arr: (nr+1, nz) -> (nz, ni)."""
    kM, _yr, f = _rate_factors_numpy(y, M, k_arr)
    rate = kM * f.prod(axis=2)
    v = rate[:, 1::2] - rate[:, 2::2]
    dy = np.zeros((y.shape[0], ni))
    np.add.at(dy, (slice(None), _RHS_SP_NP[_RHS_SEG_NP]), v[:, _RHS_PAIR_NP] * _RHS_COEF_NP)
    return dy


def chem_jac_numpy(y, M, k_arr):
    """y: (nz, ni), M: (nz,), k_arr: (nr+1, nz) -> (nz, ni, ni)."""
    kM, yr, f = _rate_factors_numpy(y, M, k_arr)
    slots = np.arange(_MAX_SLOTS)
    excl = np.stack([np.where(slots == j, 1.0, f).prod(axis=2) for j in range(_MAX_SLOTS)], axis=2)
    dpow = _R_STO_NP * yr ** np.maximum(_R_STO_NP - 1, 0)
    dr = kM[:, :, None] * dpow * excl
    J = np.zeros((y.shape[0], ni, ni))
    np.add.at(J, (slice(None), _JAC_ROWS_NP[_JAC_SEG_NP], _JAC_COLS_NP[_JAC_SEG_NP]),
              dr[:, _JAC_D_NP, _JAC_SLOT_NP] * _JAC_COEF_NP)
    return J


# ---------------------------------------------------------------------------
# Helpers and public API
# ---------------------------------------------------------------------------

def k_dict_to_array(k_dict):
    """Convert k dict {1..nr: array(nz)} to a (nr+1, nz) numpy array.

    Missing keys (e.g. photolysis reactions with photochemistry off) are zero.
    """
    nz  = len(next(iter(k_dict.values())))
    arr = np.zeros((nr + 1, nz), dtype=np.float64)
    for i, v in k_dict.items():
        arr[i] = v
    return arr


# Backend switch — True: JAX kernels (default); False: NumPy fallback.
USE_JAX_CHEM = True


def chemdf(y, M, k_dict):
    """Compute dn/dt from chemistry.

    y      : (nz, ni) numpy array
    M      : (nz,)    numpy array
    k_dict : dict {reaction_index: array(nz)}

    Returns (nz, ni) numpy array of dn/dt from chemistry.
    """
    k = k_dict_to_array(k_dict)
    if USE_JAX_CHEM:
        return np.asarray(chemdf_jax(jnp.asarray(y), jnp.asarray(M), jnp.asarray(k)))
    return chemdf_numpy(np.asarray(y), np.asarray(M), k)


def chem_jac_blocks(y, M, k_dict):
    """Return the chemistry Jacobian as a (nz, ni, ni) numpy array.

    Each jac[iz] is the positive Jacobian d(dy/dt)/dy for layer iz.
    Caller is responsible for signs and assembling into the LHS matrix.
    """
    k = k_dict_to_array(k_dict)
    if USE_JAX_CHEM:
        return np.asarray(_jac_jit(jnp.asarray(y), jnp.asarray(M), jnp.asarray(k)))
    return chem_jac_numpy(np.asarray(y), np.asarray(M), k)


def neg_achemjac(y, M, k_dict):
    """Return the negative chemistry Jacobian as a dense (ni*nz, ni*nz) NumPy array.

    Block-diagonal; diffusion coupling is added by the caller.
    """
    return _scipy_block_diag(*(-chem_jac_blocks(y, M, k_dict)))
'''


# ---------------------------------------------------------------------------
# Gibbs equilibrium-constant section
# ---------------------------------------------------------------------------

def generate_gibbs_section(reactions, gibbs_text_path):
    """Thermodynamic data block and Gibbs(i, T) equilibrium-constant function."""
    out = []
    out.append('\n\n')
    out.append('# ' + '-' * 75 + '\n')
    out.append('# Thermodynamic functions (NASA-9 polynomials)\n')
    out.append('# Included verbatim from: ' + gibbs_text_path + '\n')
    out.append('# ' + '-' * 75 + '\n\n')

    with open(gibbs_text_path) as f:
        out.append(f.read())

    out.append('\n\n')
    out.append('def Gibbs(i, T):\n')
    out.append('    """Return K_eq for forward reaction i at temperature T.\n\n')
    out.append('    Derived from NASA-9 Gibbs free energies; units consistent with rate coefficients.\n')
    out.append('    """\n')
    out.append('    G = {}\n')

    for j, reac, prod, _rxn_str in reactions:
        reac_noM = [(stoi, name) for stoi, name in reac if name != 'M']
        prod_noM = [(stoi, name) for stoi, name in prod if name != 'M']
        reac_num = sum(stoi for stoi, _ in reac_noM)
        prod_num = sum(stoi for stoi, _ in prod_noM)

        expr = 'np.exp( -('
        for stoi, name in reac_noM:
            expr += f"-{stoi}*gibbs_sp('{name}',T)"
        for stoi, name in prod_noM:
            expr += f"+{stoi}*gibbs_sp('{name}',T)"
        expr += ' ) )'
        if prod_num - reac_num != 0:
            expr += f'*(corr*T)**{reac_num - prod_num}'

        out.append(f'    G[{j}] = lambda T: {expr}\n')

    out.append('    return G[i](T)\n')

    return ''.join(out)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(chem_dict, reactions, ofname, network_name, gibbs_text_path):
    t = build_tables(chem_dict, reactions)
    ni, nr = t['ni'], t['nr']
    idx_to_name = {v: k for k, v in chem_dict.items()}
    spec_list = [idx_to_name[i] for i in range(ni)]

    out = [_HEADER.format(network=network_name, ni=ni, nr=nr, spec_list=spec_list,
                          max_slots=t['max_slots'], max_stoi=t['max_stoi'],
                          max_mpow=t['max_mpow'])]

    out.append('# Reactant species index / stoichiometry per direction and slot; pad -> (ni, 0).\n')
    out.append(_emit_array('_R_IDX_NP', t['r_idx']))
    out.append(_emit_array('_R_STO_NP', t['r_sto']))
    out.append('# Third-body (M) power per direction.\n')
    out.append(_emit_array('_M_POW_NP', t['m_pow']))
    out.append('# RHS scatter list over reversible pairs (pair p = directions 2p+1, 2p+2), sorted by receiving species.\n')
    out.append(_emit_array('_RHS_PAIR_NP', t['rhs']['pair']))
    out.append(_emit_array('_RHS_COEF_NP', t['rhs']['coef']))
    out.append(_emit_array('_RHS_SEG_NP', t['rhs']['seg']))
    out.append(_emit_array('_RHS_SP_NP', t['rhs']['sp']))
    out.append('# Jacobian scatter list, sorted by (row, col) of the non-zero entries.\n')
    out.append(_emit_array('_JAC_D_NP', t['jac']['d']))
    out.append(_emit_array('_JAC_SLOT_NP', t['jac']['slot']))
    out.append(_emit_array('_JAC_COEF_NP', t['jac']['coef']))
    out.append(_emit_array('_JAC_SEG_NP', t['jac']['seg']))
    out.append(_emit_array('_JAC_ROWS_NP', t['jac']['rows']))
    out.append(_emit_array('_JAC_COLS_NP', t['jac']['cols']))

    smax = max(t['max_stoi'], t['max_mpow'])
    out.append(_KERNELS.replace('{int_pow}', _emit_int_pow('_int_pow', smax)))
    out.append(generate_gibbs_section(reactions, gibbs_text_path))

    with open(ofname, 'w') as f:
        f.write(''.join(out))

    print(f"Wrote {ofname}")
    print(f"  {ni} species, {nr} reactions (fwd+rev)")
    print(f"  {t['jac']['rows'].shape[0]} non-zero Jacobian (i,r) pairs out of {ni*ni}")
    print(f"  Network: {network_name}")


if __name__ == '__main__':
    _argp = argparse.ArgumentParser(description='Generate src/chemistry_jax.py from a VULCAN network file.')
    _argp.add_argument('-c', '--config', default='vulcan_cfg.toml',
                       help='Path to TOML config file (default: vulcan_cfg.toml in the neoVULCAN root)')
    _args = _argp.parse_args()

    from neovulcan_config import VulcanConfig
    from neovulcan_runtime import set_cfg, get_cfg
    _cfg_path = _args.config if os.path.isabs(_args.config) else os.path.join(_root, _args.config)
    set_cfg(VulcanConfig.from_toml(_cfg_path, base_dir=_root))
    cfg = get_cfg()

    print(f"Parsing network: {cfg.network.network}")
    chem_dict, reactions = parse_network(cfg.network.network)
    generate(chem_dict, reactions, _OFNAME, cfg.network.network, cfg.network.gibbs_text)
