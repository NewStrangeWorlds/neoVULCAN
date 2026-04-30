#!/usr/bin/env python3
"""
make_chemistry_jax.py — generates src/chemistry_jax.py from a VULCAN network file.

Usage:
    python make_chemistry_jax.py

Reads the network specified in vulcan_cfg.network and writes
src/chemistry_jax.py, which provides NumPy-native drop-in replacements for
chem_funs.chemdf and chem_funs.neg_symjac, plus JAX versions as fallback.

Run this whenever the chemical network changes, before running vulcan.py.
The workflow mirrors make_chem_funs.py:

    python make_chem_funs.py      # renumber network file + generate chem_funs.py
    python make_chemistry_jax.py  # generate chemistry_jax.py from same network
    python vulcan.py              # run simulation
"""

import os
import sys
from collections import defaultdict

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, 'src'))
import vulcan_cfg

_OFNAME = os.path.join(_root, 'src', 'chemistry_jax.py')


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


# ---------------------------------------------------------------------------
# Expression builders — shared between JAX and NumPy generators
# ---------------------------------------------------------------------------

def _rate_expr(species_list, k_idx, chem_dict, numpy_mode=False):
    """Rate expression string.  numpy_mode uses y[:,idx] for layer vectorisation."""
    s = f"k[{k_idx}]"
    for stoi, name in species_list:
        if name == 'M':
            s += "*M" if stoi == 1 else f"*M**{stoi}"
        else:
            idx = chem_dict[name]
            ref = f"y[:,{idx}]" if numpy_mode else f"y[{idx}]"
            s += f"*{ref}" if stoi == 1 else f"*{ref}**{stoi}"
    return s


def build_species_terms(chem_dict, reactions, numpy_mode=False):
    """Return {species_idx: [(term_str, j, rxn_str), ...]} for every species."""
    ni = len(chem_dict)
    terms = {i: [] for i in range(ni)}

    for j, reac, prod, rxn_str in reactions:
        fwd = _rate_expr(reac, j,   chem_dict, numpy_mode)
        rev = _rate_expr(prod, j+1, chem_dict, numpy_mode)
        v_exp = f"{fwd} - {rev}"

        reac_noM = [(s, n) for s, n in reac if n != 'M']
        prod_noM = [(s, n) for s, n in prod if n != 'M']

        for stoi, name in reac_noM:
            idx = chem_dict[name]
            t = f"-({v_exp})" if stoi == 1 else f"-{stoi}*({v_exp})"
            terms[idx].append((t, j, rxn_str))

        for stoi, name in prod_noM:
            idx = chem_dict[name]
            t = f"+({v_exp})" if stoi == 1 else f"+{stoi}*({v_exp})"
            terms[idx].append((t, j, rxn_str))

    return terms


# ---------------------------------------------------------------------------
# Analytical Jacobian builder (NumPy, vectorised over layers)
# ---------------------------------------------------------------------------

def _merge_species(species_noM, chem_dict):
    """Sum stoichiometries for species that appear more than once (e.g. OH + OH -> 2*OH)."""
    merged = {}
    for stoi, name in species_noM:
        idx = chem_dict[name]
        merged[idx] = merged.get(idx, [0, name])
        merged[idx][0] += stoi
    return [(s, n) for idx, (s, n) in sorted(merged.items())]


def _deriv_rate_expr_numpy(species_noM, k_idx, wrt_idx, chem_dict, M_power):
    """NumPy expression for d(rate)/d(y[:,wrt_idx]), vectorised over layers.

    species_noM : [(stoi, name), ...] with M already excluded
    k_idx       : key into the k dict for this direction
    wrt_idx     : species index being differentiated
    M_power     : total M stoichiometry in this rate term

    Returns None when wrt_idx does not appear in species_noM.

    Duplicate species entries (e.g. two OH with stoi=1 each) are merged so
    that d(k*y_OH*y_OH)/d(y_OH) = 2*k*y_OH is computed correctly.
    """
    merged = _merge_species(species_noM, chem_dict)

    wrt_stoi = 0
    for stoi, name in merged:
        if chem_dict[name] == wrt_idx:
            wrt_stoi = stoi
            break
    if wrt_stoi == 0:
        return None

    parts = [f"k[{k_idx}]"]
    if M_power == 1:
        parts.append("M")
    elif M_power > 1:
        parts.append(f"M**{M_power}")
    if wrt_stoi > 1:
        parts.append(str(wrt_stoi))

    for stoi, name in merged:
        idx = chem_dict[name]
        eff = (stoi - 1) if idx == wrt_idx else stoi
        if eff == 1:
            parts.append(f"y[:,{idx}]")
        elif eff > 1:
            parts.append(f"y[:,{idx}]**{eff}")
        # eff == 0 → factor of 1, omit

    return "*".join(parts)


def build_jac_terms(chem_dict, reactions):
    """Return {(i, r): [(term_str, j, rxn_str), ...]} for the analytical Jacobian.

    J[iz, i, r] = d(dy_i/dt)/d(y_r).

    Sign conventions (for net velocity v = v_fwd - v_rev):
      Forward (r in reactants):  J[reactant_a, r] -= dv_fwd/dy_r
                                 J[product_b,  r] += dv_fwd/dy_r
      Reverse (r in products):   J[reactant_a, r] += dv_rev/dy_r
                                 J[product_b,  r] -= dv_rev/dy_r
    """
    jac = defaultdict(list)

    for j, reac, prod, rxn_str in reactions:
        reac_noM = [(s, n) for s, n in reac if n != 'M']
        prod_noM = [(s, n) for s, n in prod if n != 'M']
        m_fwd = sum(s for s, n in reac if n == 'M')
        m_rev = sum(s for s, n in prod if n == 'M')

        # Merge duplicate species (e.g. OH + OH → [(2,'OH')]) for correct stoichiometry
        reac_merged = _merge_species(reac_noM, chem_dict)
        prod_merged = _merge_species(prod_noM, chem_dict)

        # Forward contributions — differentiate w.r.t. each (merged) reactant
        for _sr, r_name in reac_merged:
            r_idx = chem_dict[r_name]
            dv = _deriv_rate_expr_numpy(reac_noM, j, r_idx, chem_dict, m_fwd)
            if dv is None:
                continue
            for s_a, a_name in reac_merged:
                a_idx = chem_dict[a_name]
                pre = f"-{s_a}*" if s_a > 1 else "-"
                jac[(a_idx, r_idx)].append((f"{pre}({dv})", j, rxn_str))
            for s_b, b_name in prod_merged:
                b_idx = chem_dict[b_name]
                pre = f"+{s_b}*" if s_b > 1 else "+"
                jac[(b_idx, r_idx)].append((f"{pre}({dv})", j, rxn_str))

        # Reverse contributions — differentiate w.r.t. each (merged) product
        for _sr, r_name in prod_merged:
            r_idx = chem_dict[r_name]
            dv = _deriv_rate_expr_numpy(prod_noM, j+1, r_idx, chem_dict, m_rev)
            if dv is None:
                continue
            for s_a, a_name in reac_merged:
                a_idx = chem_dict[a_name]
                pre = f"+{s_a}*" if s_a > 1 else "+"
                jac[(a_idx, r_idx)].append((f"{pre}({dv})", j, rxn_str))
            for s_b, b_name in prod_merged:
                b_idx = chem_dict[b_name]
                pre = f"-{s_b}*" if s_b > 1 else "-"
                jac[(b_idx, r_idx)].append((f"{pre}({dv})", j, rxn_str))

    return dict(jac)


# ---------------------------------------------------------------------------
# File generator — templates
# ---------------------------------------------------------------------------

_HEADER = '''\
"""Chemistry functions for neoVULCAN.

AUTO-GENERATED by make_chemistry_jax.py — do not edit by hand.
Network : {network}
Species : {ni}    Reactions (fwd+rev): {nr}

Public API (NumPy, no JAX overhead):
    chemdf(y, M, k_dict)       -> (nz, ni)       dn/dt from chemistry
    chem_jac_blocks(y, M, k_dict) -> (nz, ni, ni) positive Jacobian
    neg_achemjac(y, M, k_dict) -> (ni*nz, ni*nz) negative block-diag Jacobian

JAX fallback (for future GPU use):  chemdf_jax, _jac_jit
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import block_diag as _scipy_block_diag

# Enable 64-bit floats (JAX defaults to float32; VULCAN uses float64 throughout).
jax.config.update("jax_enable_x64", True)
# Force JAX onto CPU for single-run workloads.
jax.config.update("jax_default_device", jax.devices("cpu")[0])


# ---------------------------------------------------------------------------
# JAX single-layer chemistry:  y(ni,), M(scalar), k(nr+1,)  ->  dydt(ni,)
# ---------------------------------------------------------------------------
# k is 1-indexed (k[0] unused); y is 0-indexed (y[0]..y[{ni_1}]).

def _chemdf_single(y, M, k):
    dy = jnp.stack([
'''

# Closes _chemdf_single
_CHEMDF_CLOSE = '''\
    ])
    return dy

'''

# JAX infrastructure + public NumPy APIs (appended after NumPy section)
_POSTAMBLE = '''\
# ---------------------------------------------------------------------------
# JAX JIT-compiled vmapped versions (available as fallback / GPU path)
# ---------------------------------------------------------------------------

_chemdf_vmap = jax.vmap(_chemdf_single, in_axes=(0, 0, 1))
chemdf_jax   = jax.jit(_chemdf_vmap)

_jac_single = jax.jacfwd(_chemdf_single, argnums=0)
_jac_vmap   = jax.vmap(_jac_single, in_axes=(0, 0, 1))
_jac_jit    = jax.jit(_jac_vmap)


# ---------------------------------------------------------------------------
# Helper (used by JAX path)
# ---------------------------------------------------------------------------

def k_dict_to_array(k_dict):
    """Convert k dict {1..nr: array(nz)} to a (nr+1, nz) numpy array."""
    nr  = max(k_dict.keys())
    nz  = len(next(iter(k_dict.values())))
    arr = np.zeros((nr + 1, nz), dtype=np.float64)
    for i, v in k_dict.items():
        arr[i] = v
    return arr


# ---------------------------------------------------------------------------
# Backend switch — set to True to use JAX (auto-diff), False for analytical NumPy
# ---------------------------------------------------------------------------

USE_JAX_CHEM = True


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------

def _k_safe(k_dict, nz):
    """Return a defaultdict that yields zero arrays for missing reaction keys.

    Photolysis rates are absent when photochemistry is disabled; this avoids
    KeyError in chemdf_numpy / chem_jac_numpy for those reactions.
    """
    from collections import defaultdict
    _zero = np.zeros(nz)
    k = defaultdict(lambda: _zero)
    k.update(k_dict)
    return k


def chemdf(y, M, k_dict):
    """Drop-in for chem_funs.chemdf(y, M, k).

    y      : (nz, ni) numpy array
    M      : (nz,)    numpy array
    k_dict : dict {reaction_index: array(nz)}

    Returns (nz, ni) numpy array of dn/dt from chemistry.
    Backend selected by USE_JAX_CHEM.
    """
    if USE_JAX_CHEM:
        k = k_dict_to_array(k_dict)
        return np.asarray(chemdf_jax(jnp.asarray(y), jnp.asarray(M), jnp.asarray(k)))
    return chemdf_numpy(y, M, _k_safe(k_dict, y.shape[0]))


def chem_jac_blocks(y, M, k_dict):
    """Return the chemistry Jacobian as a (nz, ni, ni) numpy array.

    Each jac[iz] is the positive Jacobian d(dy/dt)/dy for layer iz.
    Caller is responsible for signs and assembling into the LHS matrix.
    Backend selected by USE_JAX_CHEM.
    """
    if USE_JAX_CHEM:
        k = k_dict_to_array(k_dict)
        return np.asarray(_jac_jit(jnp.asarray(y), jnp.asarray(M), jnp.asarray(k)))
    return chem_jac_numpy(y, M, _k_safe(k_dict, y.shape[0]))


def neg_achemjac(y, M, k_dict):
    """Drop-in for chem_funs.neg_symjac(y, M, k).

    Returns the *negative* chemistry Jacobian as a dense (ni*nz, ni*nz)
    NumPy array (block-diagonal; diffusion coupling added by the caller).
    Backend selected by USE_JAX_CHEM.
    """
    return _scipy_block_diag(*(-chem_jac_blocks(y, M, k_dict)))
'''



# ---------------------------------------------------------------------------
# NumPy section generator
# ---------------------------------------------------------------------------

def generate_numpy_section(chem_dict, reactions, ni, idx_to_name):
    """Emit chemdf_numpy and chem_jac_numpy source as a string."""
    out = []
    INDENT = '        '  # 8 spaces inside the expression parens

    # ---- chemdf_numpy --------------------------------------------------------
    sp_terms_np = build_species_terms(chem_dict, reactions, numpy_mode=True)

    out.append('\n\n')
    out.append('# ' + '-'*75 + '\n')
    out.append('# NumPy analytical chemistry (vectorised over all layers, no JAX overhead)\n')
    out.append('# ' + '-'*75 + '\n\n')
    out.append('def chemdf_numpy(y, M, k):\n')
    out.append('    """y: (nz,ni), M: (nz,), k: {j: array(nz)} -> (nz,ni)."""\n')
    out.append('    dy = np.empty_like(y)\n')

    for sp_idx in range(ni):
        name = idx_to_name[sp_idx]
        terms = sp_terms_np[sp_idx]
        out.append(f'    # {name} ({sp_idx})\n')
        if not terms:
            out.append(f'    dy[:,{sp_idx}] = 0.\n')
        else:
            out.append(f'    dy[:,{sp_idx}] = (\n')
            for term, j, rxn in terms:
                out.append(f'{INDENT}{term}  # R{j}: {rxn}\n')
            out.append(f'    )\n')

    out.append('    return dy\n')

    # ---- chem_jac_numpy ------------------------------------------------------
    jac_terms = build_jac_terms(chem_dict, reactions)

    out.append('\n\n')
    out.append(f'def chem_jac_numpy(y, M, k):\n')
    out.append(f'    """y: (nz,ni), M: (nz,), k: {{j: array(nz)}} -> (nz,{ni},{ni}) Jacobian.\n')
    out.append(f'    J[iz,i,r] = d(dy_i/dt)/d(y_r) for layer iz.\n')
    out.append(f'    """\n')
    out.append(f'    J = np.zeros((y.shape[0], {ni}, {ni}))\n')

    for (i, r) in sorted(jac_terms.keys()):
        terms = jac_terms[(i, r)]
        if not terms:
            continue
        out.append(f'    # d({idx_to_name[i]})/d({idx_to_name[r]})\n')
        out.append(f'    J[:,{i},{r}] = (\n')
        for term, j, rxn in terms:
            out.append(f'{INDENT}{term}  # R{j}: {rxn}\n')
        out.append(f'    )\n')

    out.append('    return J\n\n')

    return ''.join(out)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(chem_dict, reactions, ofname):
    ni = len(chem_dict)
    nr = 2 * len(reactions)
    idx_to_name = {v: k for k, v in chem_dict.items()}
    sp_terms = build_species_terms(chem_dict, reactions)

    out = []
    out.append(_HEADER.format(
        network=vulcan_cfg.network,
        ni=ni,
        nr=nr,
        ni_1=ni - 1,
    ))

    # JAX _chemdf_single body (unchanged — uses y[idx] single-layer indexing)
    INDENT = '        '
    for sp_idx in range(ni):
        name = idx_to_name[sp_idx]
        term_list = sp_terms[sp_idx]
        out.append(f'{INDENT}# {name} ({sp_idx})\n')

        if not term_list:
            out.append(f'{INDENT}jnp.array(0.),\n')
        else:
            for k_idx, (term, j, rxn_str) in enumerate(term_list):
                is_last = k_idx == len(term_list) - 1
                suffix = ',' if is_last else ''
                out.append(f'{INDENT}{term}{suffix}  # R{j}: {rxn_str}\n')

    # Close _chemdf_single, emit NumPy section, then JAX infrastructure + APIs
    out.append(_CHEMDF_CLOSE)
    out.append(generate_numpy_section(chem_dict, reactions, ni, idx_to_name))
    out.append(_POSTAMBLE)

    content = ''.join(out)
    with open(ofname, 'w') as f:
        f.write(content)

    # Count non-zero Jacobian entries
    jac_terms = build_jac_terms(chem_dict, reactions)
    print(f"Wrote {ofname}")
    print(f"  {ni} species, {nr} reactions (fwd+rev)")
    print(f"  {len(jac_terms)} non-zero Jacobian (i,r) pairs out of {ni*ni}")
    print(f"  Network: {vulcan_cfg.network}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"Parsing network: {vulcan_cfg.network}")
    chem_dict, reactions = parse_network(vulcan_cfg.network)
    generate(chem_dict, reactions, _OFNAME)
