# neoVULCAN

#### Author: Daniel Kitzmann
#### Original VULCAN authors: Shang-Min (Shami) Tsai

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

neoVULCAN is a fork of [VULCAN](https://github.com/shami-EEG/VULCAN) — a photochemical kinetics
code for exoplanetary atmospheres. This fork tracks work in progress: a JAX-based chemistry
kernel, alternative Rosenbrock solvers, and a typed TOML-based configuration system.

The theory papers of VULCAN are
[Tsai et al. 2017](https://arxiv.org/abs/1607.00409) (thermal chemistry) and
[Tsai et al. 2021](https://arxiv.org/abs/2108.01790) (with photochemistry).
Chemical equilibrium initialisation uses [FastChem](https://github.com/exoclime/FastChem).


## Installation

Requires **Python ≥ 3.10**. Install dependencies into the environment of your choice:

```bash
pip install -r requirements.txt
```

The required packages are NumPy, SciPy, JAX, pyfastchem, Matplotlib, Pydantic, Pillow (optional),
and Sympy. On Python 3.10 the TOML loader uses the `tomli` backport; on 3.11+ it uses the
stdlib `tomllib`. For GPU acceleration, install the CUDA variant of JAX (see the comment in
[requirements.txt](requirements.txt)).


## Quick start

Edit [vulcan_cfg.toml](vulcan_cfg.toml) (or copy one of the example configs from
[cfg_examples/](cfg_examples/)), then run:

```bash
python vulcan.py -c vulcan_cfg.toml
```

Both arguments accept defaults: `-c vulcan_cfg.toml` is implicit, and the flag `-n` skips
regeneration of the auto-generated chemistry module (use when the network hasn't changed
since the last run).

Output is written to `output/<out_name>.vul` (a pickle file) plus a JSON dump of the
fully-resolved configuration alongside it.

To produce plots from a `.vul` file, see [plot_py/](plot_py/).


## Configuration

Runs are configured through TOML files validated against a Pydantic schema
([src/neovulcan_config.py](src/neovulcan_config.py)). The schema groups fields into ten sections:

| Section | What it controls |
|---|---|
| `network` | Atom list, path to the reaction network, photo-cross-section directory |
| `paths` | All filesystem paths for input/output and FastChem |
| `elements` | Elemental abundances or solar mix; initial mixing-ratio strategy |
| `atmosphere` | Vertical grid, pressure range, T-P profile source, gravity, stellar geometry |
| `photochemistry` | Photolysis switches, RT scheme, wavelength binning |
| `boundary_conditions` | Top/bottom fluxes, fixed mixing ratios, diffusion-limited escape |
| `condensation` | Condensible species, particle properties, relaxation |
| `solver` | ODE solver choice, tolerances, step controls, Newton finisher knobs |
| `output` | What to save to the `.vul` file |
| `plotting` | Live plots, movies, species to track |

The schema rejects unknown keys (catches typos), validates enum values (e.g.
`atm_type ∈ {"isothermal", "analytical", "file", "vulcan_ini", "table"}`), and enforces
cross-field constraints (`P_t < P_b`, `use_ion ⇒ use_photo`, etc.). Several fields have
sensible defaults — omit them from the TOML to use the default.

To see every available field with its default value and type, look at
[vulcan_cfg_defaults.toml](vulcan_cfg_defaults.toml). That file is generated from the
schema via:

```bash
python tests/_gen_defaults_toml.py
```

Example configurations are in [cfg_examples/](cfg_examples/):
- `Earth.cfg` — Earth-like atmosphere (N2/O2, photochemistry, water + sulfate condensation)
- `Jupiter.cfg` — Jupiter (H2 atmosphere, water and ammonia condensation, NCHO low-T network)
- `HD189.cfg` — HD 189733b hot Jupiter (SNCHO network with sulfur and helium)


## Library mode

For external models that want to drive neoVULCAN as a chemistry step:

```python
import sys
sys.path.insert(0, '/path/to/VULCAN/neoVULCAN')
from neoVULCAN import VulcanChemistry

chem = VulcanChemistry(
    base_dir='/path/to/VULCAN/neoVULCAN',
    config_path='vulcan_cfg.toml',        # default — pass a different path to switch configs
    cfg_overrides={'solver': {'rtol': 0.1}},   # optional nested-dict overrides
)
chem.initialize()

for step in range(n_steps):
    T_new, P_new, Kzz_new = atm_model.get_profiles()
    chem.set_atmosphere(T=T_new, P=P_new, Kzz=Kzz_new)
    chem.run_to_convergence()
    ymix = chem.get_mixing_ratios()       # (nz, ni)
    atm_model.update_chemistry(ymix, chem.species)
```

See [vulcan_api.py](vulcan_api.py) for the full API.


## Repository layout

```
neoVULCAN/
├── vulcan.py                # script entry point
├── vulcan_api.py            # library entry point (VulcanChemistry)
├── __init__.py              # re-exports VulcanChemistry
├── vulcan_cfg.toml          # active configuration
├── vulcan_cfg_defaults.toml # reference: every field, its type, its default
│
├── src/                     # library modules
│   ├── neovulcan_config.py    Pydantic schema (VulcanConfig)
│   ├── neovulcan_runtime.py   process-singleton holding the loaded config
│   ├── make_chemistry_jax.py  generator: produces chemistry_jax.py from a network file
│   ├── chemistry_jax.py       auto-generated; do not edit by hand
│   ├── integration.py         time-stepping loop
│   ├── ros2.py, rodas3.py     Rosenbrock ODE solvers
│   ├── ode_solver.py          base ODE machinery, banded Jacobian assembly
│   ├── build_atm.py, store.py atmospheric setup, data containers
│   ├── rates.py               reaction-rate reader
│   ├── radiative_transfer.py  two-stream / DisORT photochemistry
│   ├── condensation.py        condensation kinetics
│   ├── output.py              .vul file writer, plotting helpers
│   ├── phy_const.py           physical constants
│   └── jacobian_jax.py        JAX-fused Jacobian kernel
│
├── tests/                   # pytest tests + benchmarks + dev tools
├── plot_py/                 # plotting scripts for .vul output
├── tools/                   # miscellaneous utilities
├── cfg_examples/            # example .cfg files (Earth, Jupiter, HD189)
├── atm/                     # T-P-Kzz profiles, stellar fluxes, boundary-condition files
├── thermo/                  # reaction networks, NASA-9 thermodata, photolysis cross-sections
├── fastchem_input/          # FastChem equilibrium-chemistry input
├── output/                  # default .vul output directory
└── plot/                    # default plot directory
```


## Running tests

```bash
pytest tests/                    # full suite
pytest tests/ -m "not slow"      # skip slow regression tests
```

The migration-time parity tests in [tests/test_config_migration.py](tests/test_config_migration.py)
also cover Pydantic schema correctness (typo rejection, alias semantics, cross-field validators).
The radiative-transfer snapshot test ([tests/test_rt_refactor.py](tests/test_rt_refactor.py))
auto-skips if its `rt_snapshot.pkl` was captured against a different network; regenerate via
`python tests/capture_rt_snapshot.py`.


## Licence

GPL v3 — see [licence.md](licence.md).
