#!/usr/bin/env python3
"""
make_fastchem_input.py
Convert VULCAN's old NASA-9 logK files (nasa9_logK_*.dat) into the pyfastchem
5-coefficient format.

Reads:
  fastchem_vulcan/input/nasa9_logK_SNCHOPTi.dat   → logK_vulcan.dat
  fastchem_vulcan/input/nasa9_logK_SNCHOTi_ion.dat → logK_vulcan_ion.dat

Output files are placed in fastchem_vulcan/input/.

Run from: neoVULCAN/
Re-run whenever the nasa9_logK_*.dat source files change.

Algorithm
---------
The old C++ FastChem read 20 raw NASA-9 coefficients per species (10 low-T,
10 high-T) and computed at runtime:

  lnK_molecule(T) = a[0]/(2T²) - a[1]*(1+lnT)/T - a[2]*(1-lnT)
                  + a[3]*T/2 + a[4]*T²/6 + a[5]*T³/12 + a[6]*T⁴/20
                  - a[8]/T + a[9]

It then subtracted the same quantity evaluated for each constituent element
(using hardcoded coefficients from mass_action_constant.cpp) to obtain the
formation lnK:

  lnK_formation = lnK_molecule - Σ ν_j * lnK_element_j

pyfastchem uses a 5-coefficient analytic fit:

  lnK(T) = c[0]/T + c[1]*lnT + c[2] + c[3]*T + c[4]*T²

We reproduce this computation and fit the 5-coefficient model over
T = 200 … 6000 K, then write the result in pyfastchem's logK format.
"""

import numpy as np
import os
import re

# Regex for floating-point numbers, including Fortran-style adjacent negatives
# e.g. '3.17E-06-1.79E-09' → ['3.17E-06', '-1.79E-09']
_FLOAT_RE = re.compile(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?')

# ── Element NASA-9 coefficients (from mass_action_constant.cpp) ─────────────
# Each entry: {'low': [a0..a9], 'high': [a0..a9]}
# lnK formula:  a[0]/(2T²) - a[1]*(1+lnT)/T - a[2]*(1-lnT)
#             + a[3]*T/2 + a[4]*T²/6 + a[5]*T³/12 + a[6]*T⁴/20
#             - a[8]/T + a[9]       (a[7] is always 0)

ELEMENT_COEFFS = {
    'C': {
        'low':  [ 6.495031470E+02, -9.649010860E-01,  2.504675479E+00, -1.281448025E-05,
                  1.980133654E-08, -1.606144025E-11,  5.314483411E-15, 0,
                  8.545763110E+04,  4.747924288E+00],
        'high': [-1.289136472E+05,  1.719528572E+02,  2.646044387E+00, -3.353068950E-04,
                  1.742092740E-07, -2.902817829E-11,  1.642182385E-15, 0,
                  8.410597850E+04,  4.130047418E+00],
    },
    'H': {
        'low':  [ 0,  0,  2.5,  0,  0,  0,  0, 0,
                  2.547370801E+04, -4.466828530E-01],
        'high': [ 6.078774250E+01, -1.819354417E-01,  2.500211817E+00, -1.226512864E-07,
                  3.732876330E-11, -5.687744560E-15,  3.410210197E-19, 0,
                  2.547486398E+04, -4.481917770E-01],
    },
    'N': {
        'low':  [ 0,  0,  2.5,  0,  0,  0,  0, 0,
                  5.610463780E+04,  4.193905036E+00],
        'high': [ 8.876501380E+04, -1.071231500E+02,  2.362188287E+00,  2.916720081E-04,
                 -1.729515100E-07,  4.012657880E-11, -2.677227571E-15, 0,
                  5.697351330E+04,  4.865231506E+00],
    },
    'O': {
        'low':  [-7.953611300E+03,  1.607177787E+02,  1.966226438E+00,  1.013670310E-03,
                 -1.110415423E-06,  6.517507500E-10, -1.584779251E-13, 0,
                  2.840362437E+04,  8.404241820E+00],
        'high': [ 2.619020262E+05, -7.298722030E+02,  3.317177270E+00, -4.281334360E-04,
                  1.036104594E-07, -9.438304330E-12,  2.725038297E-16, 0,
                  3.392428060E+04, -6.679585350E-01],
    },
    'He': {
        'low':  [ 0,  0,  2.5,  0,  0,  0,  0, 0,
                 -7.453750000E+02,  9.287239740E-01],
        'high': [ 0,  0,  2.5,  0,  0,  0,  0, 0,
                 -7.453750000E+02,  9.287239740E-01],
    },
    'S': {
        'low':  [-3.174841820E+02, -1.924704923E+02,  4.686825930E+00, -5.841365600E-03,
                  7.538533520E-06, -4.863586040E-09,  1.256976992E-12, 0,
                  3.323592180E+04, -5.718523969E+00],
        'high': [-4.854244790E+05,  1.438830408E+03,  1.258504116E+00,  3.797990430E-04,
                  1.630685864E-09, -9.547095850E-12,  8.041466646E-16, 0,
                  2.334995270E+04,  1.559554855E+01],
    },
    'P': {
        'low':  [ 5.040866570E+01, -7.639418650E-01,  2.504563992E+00, -1.381689958E-05,
                  2.245585515E-08, -1.866399889E-11,  6.227063395E-15, 0,
                  3.732421910E+04,  5.359303481E+00],
        'high': [ 1.261794642E+06, -4.559838190E+03,  8.918079310E+00, -4.381401460E-03,
                  1.454286224E-06, -2.030782763E-10,  1.021022887E-14, 0,
                  6.541723960E+04, -3.915974795E+01],
    },
    'Na': {
        'low':  [ 0,  0,  2.5,  0,  0,  0,  0, 0,
                  1.218382949E+04,  4.244028180E+00],
        'high': [ 9.525723380E+05, -2.623807254E+03,  5.162596620E+00, -1.210218586E-03,
                  2.306301844E-07, -1.249597843E-11,  7.226771190E-16, 0,
                  2.912963564E+04, -1.519717061E+01],
    },
    'K': {
        'low':  [ 9.665143930E+00, -1.458059455E-01,  2.500865861E+00, -2.601219276E-06,
                  4.187306580E-09, -3.439722110E-12,  1.131569009E-15, 0,
                  9.959493490E+03,  5.035822260E+00],
        'high': [-3.566422360E+06,  1.085289825E+04, -1.054134898E+01,  8.009801350E-03,
                 -2.696681041E-06,  4.715294150E-10, -2.976897350E-14, 0,
                 -5.875337010E+04,  9.738551240E+01],
    },
    'Si': {
        'low':  [ 9.836140810E+01,  1.546544523E+02,  1.876436670E+00,  1.320637995E-03,
                 -1.529720059E-06,  8.950562770E-10, -1.952873490E-13, 0,
                  5.263510310E+04,  9.698288880E+00],
        'high': [-6.169298850E+05,  2.240683927E+03, -4.448619320E-01,  1.710056321E-03,
                 -4.107714160E-07,  4.558884780E-11, -1.889515353E-15, 0,
                  3.953558760E+04,  2.679668061E+01],
    },
    'Fe': {
        'low':  [ 6.790822660E+04, -1.197218407E+03,  9.843393310E+00, -1.652324828E-02,
                  1.917939959E-05, -1.149825371E-08,  2.832773807E-12, 0,
                  5.466995940E+04, -3.383946260E+01],
        'high': [-1.954923682E+06,  6.737161100E+03, -5.486410970E+00,  4.378803450E-03,
                 -1.116286672E-06,  1.544348856E-10, -8.023578182E-15, 0,
                  7.137370060E+03,  6.504979860E+01],
    },
    'Ti': {
        'low':  [-4.570179400E+04,  6.608092020E+02,  4.295257490E-01,  3.615029910E-03,
                 -3.549792810E-06,  1.759952494E-09, -3.052720871E-13, 0,
                  5.270947930E+04,  2.026149738E+01],
        'high': [-1.704786714E+05,  1.073852803E+03,  1.181955014E+00,  2.245246352E-04,
                  3.091697848E-07, -5.740027280E-11,  2.927371014E-15, 0,
                  4.978069910E+04,  1.740431368E+01],
    },
    'V': {
        'low':  [-5.535376020E+04,  5.593338510E+02,  2.675543482E+00, -6.243049630E-03,
                  1.565902337E-05, -1.372845314E-08,  4.168388810E-12, 0,
                  5.820664360E+04,  9.524567490E+00],
        'high': [ 1.200390300E+06, -5.027005300E+03,  1.058830594E+01, -5.044326100E-03,
                  1.488547375E-06, -1.785922508E-10,  8.113013866E-15, 0,
                  9.170740910E+04, -4.768336320E+01],
    },
    'Mg': {
        'low':  [ 0,  0,  2.5,  0,  0,  0,  0, 0,
                  1.694658761E+04,  3.634330140E+00],
        'high': [-5.364831550E+05,  1.973709576E+03, -3.633776900E-01,  2.071795561E-03,
                 -7.738051720E-07,  1.359277788E-10, -7.766898397E-15, 0,
                  4.829188110E+03,  2.339104998E+01],
    },
    'Ca': {
        'low':  [ 0,  0,  2.5,  0,  0,  0,  0, 0,
                  2.063892786E+04,  4.384548330E+00],
        'high': [ 7.547341240E+06, -2.148642662E+04,  2.530849567E+01, -1.103773705E-02,
                  2.293249636E-06, -1.209075383E-10, -4.015333268E-15, 0,
                  1.585862323E+05, -1.609512955E+02],
    },
    'Cl': {
        'low':  [ 2.276215854E+04, -2.168413293E+02,  2.745185115E+00,  2.451101694E-03,
                 -5.458011990E-06,  4.417986880E-09, -1.288134004E-12, 0,
                  1.501357068E+04,  3.102963457E+00],
        'high': [-1.697932930E+05,  6.081726460E+02,  2.128664090E+00,  1.307367034E-04,
                 -2.644883596E-08,  2.842504775E-12, -1.252911731E-16, 0,
                  9.934387400E+03,  8.844772103E+00],
    },
    'F': {
        'low':  [ 1.137409088E+03, -1.453392797E+02,  4.077403610E+00, -4.303360140E-03,
                  5.728897740E-06, -3.819312900E-09,  1.018322509E-12, 0,
                  9.311110120E+03, -3.558982650E+00],
        'high': [ 1.473506226E+04,  8.149927360E+01,  2.444371819E+00,  2.120210026E-05,
                 -4.546918620E-09,  5.109528730E-13, -2.333894647E-17, 0,
                  8.388374650E+03,  5.478710640E+00],
    },
    'e-': {
        'low':  [ 0,  0,  2.5,  0,  0,  0,  0, 0,
                 -7.453750000E+02, -1.172081224E+01],
        'high': [ 0,  0,  2.5,  0,  0,  0,  0, 0,
                 -7.453750000E+02, -1.172081224E+01],
    },
}

# Elements tracked as atomic species by pyfastchem (must not appear in logK file)
PYFASTCHEM_ELEMENTS = set(ELEMENT_COEFFS.keys())

T_FIT = np.logspace(np.log10(200), np.log10(6000), 1000)


def nasa9_lnK(T, a):
    """Compute lnK = -G/(RT) from 10 NASA-9 coefficients a[0..9]."""
    return (a[0] / (2 * T**2)
            - a[1] * (1 + np.log(T)) / T
            - a[2] * (1 - np.log(T))
            + a[3] * T / 2
            + a[4] * T**2 / 6
            + a[5] * T**3 / 12
            + a[6] * T**4 / 20
            - a[8] / T
            + a[9])


def nasa9_lnK_vec(T_arr, a_low, a_high):
    """Vectorised nasa9_lnK over a temperature array."""
    low = T_arr <= 1000.0
    result = np.empty(len(T_arr))
    if np.any(low):
        result[low]  = nasa9_lnK(T_arr[low],  np.asarray(a_low))
    if np.any(~low):
        result[~low] = nasa9_lnK(T_arr[~low], np.asarray(a_high))
    return result


def fit_logK(lnK_values):
    """Fit 5-coefficient model c0/T + c1*lnT + c2 + c3*T + c4*T² via lstsq."""
    X = np.column_stack([1 / T_FIT, np.log(T_FIT),
                         np.ones(len(T_FIT)), T_FIT, T_FIT**2])
    coeffs, _, _, _ = np.linalg.lstsq(X, lnK_values, rcond=None)
    return coeffs


def parse_stoich(stoich_str):
    """
    Parse stoichiometry string like 'C 1 H 2 O 1' or 'H 1 e- -1'.
    Returns dict {element_symbol: count} with integer counts.
    """
    tokens = stoich_str.split()
    stoich = {}
    i = 0
    while i < len(tokens) - 1:
        el = tokens[i]
        try:
            nu = int(tokens[i + 1])
        except ValueError:
            i += 1
            continue
        stoich[el] = nu
        i += 2
    return stoich


def is_atomic_element(stoich):
    """
    True if this species is a bare atomic element tracked by pyfastchem
    (exactly one heavy element, count=1, no electrons).
    These must NOT appear in the logK file.
    """
    non_electron = {el: nu for el, nu in stoich.items() if el != 'e-'}
    if len(non_electron) == 1 and list(non_electron.values())[0] == 1:
        if stoich.get('e-', 0) == 0:
            return True
    return False


def parse_nasa9_logK_file(path):
    """
    Parse a nasa9_logK_*.dat file.

    Each entry is two logical lines:
      Symbol [Name] : El N El N ... [# comment]
      <20 NASA-9 coefficients on one line>

    Returns list of (symbol, stoich_dict, a_low, a_high).
    """
    entries = []
    with open(path, encoding='utf-8') as f:
        raw = f.read()

    # Split into non-comment, non-blank lines
    lines = [l for l in raw.splitlines()]

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip blank lines and comment lines
        if not line or line.startswith('#'):
            i += 1
            continue

        # Header line contains ' : '
        if ' : ' in line:
            # Strip trailing comment
            header = line.split('#')[0].strip()
            colon = header.index(' : ')

            before = header[:colon].split()
            symbol = before[0]
            stoich_str = header[colon + 3:].strip()
            stoich = parse_stoich(stoich_str)

            # Next non-empty line is the coefficient line
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1

            if i >= len(lines):
                break

            coeff_text = lines[i].strip()
            # Use regex extraction to handle adjacent numbers without spaces,
            # e.g. '3.17E-06-1.79E-09' (common Fortran output quirk)
            try:
                vals = [float(m) for m in _FLOAT_RE.findall(coeff_text)]
            except ValueError:
                print(f'WARNING: cannot parse coefficients for {symbol}, skipping.')
                i += 1
                continue

            if len(vals) < 20:
                print(f'WARNING: {symbol} has only {len(vals)} coefficients (need 20), skipping.')
                i += 1
                continue

            a_low  = np.array(vals[:10])
            a_high = np.array(vals[10:20])
            entries.append((symbol, stoich, a_low, a_high))

        i += 1

    return entries


def convert_file(input_path, output_path, label):
    """
    Convert one nasa9_logK_*.dat file to pyfastchem 5-coefficient format.
    """
    entries = parse_nasa9_logK_file(input_path)
    print(f'\nProcessing {os.path.basename(input_path)} ({len(entries)} entries)')

    header = (
        '#logK = a1/T + a2 ln T + a3 + a4 T + a5 T^2 for FastChem\n'
        '#Converted from NASA-9 polynomials by make_fastchem_input.py\n'
        f'#Source: {os.path.basename(input_path)}\n'
    )

    out_lines = []
    n_written = 0
    n_skipped_el = 0
    n_skipped_missing = 0
    seen = set()

    for symbol, stoich, a_low, a_high in entries:
        # Deduplicate
        if symbol in seen:
            print(f'  INFO: duplicate {symbol}, skipping second occurrence.')
            continue
        seen.add(symbol)

        # Skip bare atomic elements (tracked by pyfastchem as elements)
        if is_atomic_element(stoich):
            n_skipped_el += 1
            continue

        # Check all elements have known coefficients
        missing = [el for el in stoich if el not in ELEMENT_COEFFS]
        if missing:
            print(f'  WARNING: {symbol} contains unknown element(s) {missing}, skipping.')
            n_skipped_missing += 1
            continue

        # Compute raw molecular lnK over temperature grid
        lnK_vals = nasa9_lnK_vec(T_FIT, a_low, a_high)

        # Subtract element contributions: lnK_formation = lnK_mol - Σ ν_j * lnK_element_j
        for el, nu in stoich.items():
            ec = ELEMENT_COEFFS[el]
            lnK_vals -= nu * nasa9_lnK_vec(T_FIT, ec['low'], ec['high'])

        # Fit 5-coefficient model
        c = fit_logK(lnK_vals)

        # Build stoichiometry string for pyfastchem (same as source file)
        stoich_str = ' '.join(f'{el} {nu}' for el, nu in stoich.items())

        out_lines.append(
            f'{symbol} {symbol} : {stoich_str}\n'
            f'  {c[0]:.6e}  {c[1]:.6e}  {c[2]:.6e}  {c[3]:.6e}  {c[4]:.6e}\n\n'
        )
        n_written += 1

    with open(output_path, 'w') as f:
        f.write(header)
        f.writelines(out_lines)

    print(f'  Written  : {n_written} species → {output_path}')
    print(f'  Skipped  : {n_skipped_el} atomic elements, '
          f'{n_skipped_missing} with unknown elements')


def main():
    input_dir  = ''
    os.makedirs(input_dir, exist_ok=True)

    convert_file(
        os.path.join(input_dir, 'nasa9_logK_SNCHOPTi.dat'),
        os.path.join(input_dir, 'logK_vulcan.dat'),
        label='neutral',
    )

    convert_file(
        os.path.join(input_dir, 'nasa9_logK_SNCHOTi_ion.dat'),
        os.path.join(input_dir, 'logK_vulcan_ion.dat'),
        label='ion',
    )

    print('\nDone.')


if __name__ == '__main__':
    main()
