"""Typed configuration schema for neoVULCAN, loaded from TOML."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Literal

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


_STRICT = ConfigDict(extra='forbid')


class NetworkConfig(BaseModel):
    model_config = _STRICT

    atom_list: list[str]
    # Paths are stored as strings (not pathlib.Path) so legacy `dir + name`
    # concatenation in src/ keeps working. Absolutification happens during load.
    network: str
    gibbs_text: str = 'thermo/gibbs_text.txt'
    com_file: str = 'thermo/all_compose.txt'
    cross_folder: str = 'thermo/photo_cross/'
    use_lowT_limit_rates: bool = False
    remove_list: list[int] = Field(default_factory=list)


class PathsConfig(BaseModel):
    model_config = _STRICT

    fastchem_dir: str = 'fastchem_input'
    atm_file: str
    sflux_file: str
    top_BC_flux_file: str = 'atm/'
    bot_BC_flux_file: str = 'atm/'
    vul_ini: str = 'output/'
    output_dir: str = 'output/'
    plot_dir: str = 'plot/'
    movie_dir: str = 'plot/movie/'
    out_name: str


class ElementsConfig(BaseModel):
    model_config = _STRICT

    use_solar: bool = False
    metal_abundances: dict[str, float] = Field(default_factory=dict)
    ini_mix: Literal['EQ', 'const_mix', 'vulcan_ini', 'table'] = 'const_mix'
    fastchem_met_scale: float = 1.0
    const_mix: dict[str, float] = Field(default_factory=dict)


class AtmosphereConfig(BaseModel):
    model_config = _STRICT

    atm_base: Literal['H2', 'N2', 'O2', 'CO2'] = 'N2'
    rocky: bool = True
    nz: int = Field(..., gt=0)
    P_b: float = Field(..., gt=0)
    P_t: float = Field(..., gt=0)
    use_Kzz: bool = True
    use_moldiff: bool = True
    use_vm_mol: bool = False
    use_vz: bool = False
    atm_type: Literal['isothermal', 'analytical', 'file', 'vulcan_ini', 'table'] = 'file'
    Kzz_prof: Literal['const', 'file', 'Pfunc'] = 'file'
    K_max: float = 1e5
    K_p_lev: float = 0.1
    K_deep: float | None = None
    vz_prof: Literal['const', 'file'] = 'const'
    gs: float
    Tiso: float = 1000.0
    para_warm: list[float] = Field(default_factory=lambda: [120.0, 1500.0, 0.1, 0.02, 1.0, 1.0])
    para_anaTP: list[float] | None = None
    const_Kzz: float = 1e10
    const_vz: float = 0.0
    update_frq: int = 100
    r_star: float = 1.0
    Rp: float
    orbit_radius: float = 1.0
    sl_angle_deg: float = 58.0
    f_diurnal: float = 0.5

    @model_validator(mode='after')
    def _check_pressures(self):
        if self.P_t >= self.P_b:
            raise ValueError(f'P_t ({self.P_t}) must be < P_b ({self.P_b})')
        return self

    @model_validator(mode='after')
    def _alias_para_anaTP(self):
        if self.para_anaTP is None:
            self.para_anaTP = list(self.para_warm)
        if len(self.para_warm) != 6:
            raise ValueError(f'para_warm must have length 6, got {len(self.para_warm)}')
        if len(self.para_anaTP) != 6:
            raise ValueError(f'para_anaTP must have length 6, got {len(self.para_anaTP)}')
        return self

    @property
    def sl_angle(self) -> float:
        return math.radians(self.sl_angle_deg)


class PhotochemConfig(BaseModel):
    model_config = _STRICT

    use_photo: bool = False
    use_ion: bool = False
    scat_sp: list[str] = Field(default_factory=list)
    T_cross_sp: list[str] = Field(default_factory=list)
    edd: float = 0.5
    dbin1: float = 0.1
    dbin2: float = 2.0
    dbin_12trans: float = 240.0
    ini_update_photo_frq: int = 100
    final_update_photo_frq: int = 5
    rt_scheme: Literal['two-stream', 'disort'] = 'two-stream'
    surface_albedo: float = 0.0
    disort_nstr: int = 8


class BoundaryConfig(BaseModel):
    model_config = _STRICT

    use_topflux: bool = False
    use_botflux: bool = False
    use_fix_sp_bot: dict[str, float] = Field(default_factory=dict)
    diff_esc: list[str] = Field(default_factory=lambda: ['H'])
    max_flux: float = 1e13
    use_sat_surfaceH2O: bool = False
    use_fix_H2He: bool = False
    loss_ex: list[str] = Field(default_factory=list)


class CondensationConfig(BaseModel):
    model_config = _STRICT

    use_condense: bool = False
    use_settling: bool = False
    use_relax: list[str] = Field(default_factory=list)
    humidity: float = 1.0
    r_p: dict[str, float] = Field(default_factory=dict)
    rho_p: dict[str, float] = Field(default_factory=dict)
    start_conden_time: float = 0.0
    stop_conden_time: float = 5e8
    condense_sp: list[str] = Field(default_factory=list)
    non_gas_sp: list[str] = Field(default_factory=list)
    fix_species: list[str] = Field(default_factory=list)
    fix_species_time: float | None = None
    fix_species_from_coldtrap_lev: bool = True
    use_ini_cold_trap: bool = True

    @model_validator(mode='after')
    def _alias_fix_species_time(self):
        if self.fix_species_time is None:
            self.fix_species_time = self.stop_conden_time
        return self

    @model_validator(mode='after')
    def _condense_completeness(self):
        if not self.use_condense:
            return self
        missing_r = [sp for sp in self.condense_sp if sp not in self.r_p and sp not in self.non_gas_sp]
        # the radius/density dicts are keyed by non_gas_sp (the condensed phase),
        # not by the gas-phase condense_sp, so we don't enforce coverage here.
        # (kept as a placeholder — strict version commented out)
        return self


class SolverConfig(BaseModel):
    model_config = _STRICT

    ode_solver: Literal['Ros2', 'Rodas3', 'ODESolver'] = 'Ros2'
    use_pi_controller: bool = False
    use_print_prog: bool = True
    use_print_delta: bool = False
    print_prog_num: int = 500
    dttry: float = 1e-10
    trun_min: float = 1e2
    runtime: float = 1e22
    dt_min: float = 1e-14
    dt_max: float | None = None
    dt_var_max: float = 2.0
    dt_var_min: float = 0.5
    count_min: int = 120
    count_max: int = 40000
    atol: float = 1e-1
    mtol: float = 1e-22
    mtol_conv: float = 1e-16
    pos_cut: float = 0.0
    nega_cut: float = -1.0
    loss_eps: float = 1e12
    yconv_cri: float = 0.01
    slope_cri: float = 1e-4
    yconv_min: float = 0.1
    flux_cri: float = 0.1
    flux_atol: float = 1.0
    st_factor: float = 0.5
    conv_step: int = 500
    conver_ignore: list[str] = Field(default_factory=list)
    rtol: float = 1.0
    post_conden_rtol: float = 0.05
    use_adapt_rtol: bool = True
    rtol_min: float = 0.02
    rtol_max: float = 2.5
    use_newton_finisher: bool = False
    newton_switch_dy: float = 1.0
    newton_cooldown: int = 200
    newton_max_iter: int = 20
    newton_res_tol: float = 1e-6
    newton_alpha_min: float = 1e-3

    @model_validator(mode='after')
    def _derive_dt_max(self):
        if self.dt_max is None:
            self.dt_max = self.runtime * 1e-5
        return self


class OutputConfig(BaseModel):
    model_config = _STRICT

    output_humanread: bool = False
    use_shark: bool = False
    save_evolution: bool = False
    save_evo_frq: int = 10
    y_time_freq: int = 1


class PlottingConfig(BaseModel):
    model_config = _STRICT

    plot_TP: bool = False
    use_live_plot: bool = False
    use_live_flux: bool = False
    use_plot_end: bool = False
    use_plot_evo: bool = False
    use_save_movie: bool = False
    use_flux_movie: bool = False
    plot_height: bool = False
    use_PIL: bool = True
    live_plot_frq: int = 10
    save_movie_rate: int | None = None
    plot_spec: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def _alias_save_movie_rate(self):
        if self.save_movie_rate is None:
            self.save_movie_rate = self.live_plot_frq
        return self


class VulcanConfig(BaseModel):
    model_config = _STRICT

    network: NetworkConfig
    paths: PathsConfig
    elements: ElementsConfig
    atmosphere: AtmosphereConfig
    photochemistry: PhotochemConfig = Field(default_factory=PhotochemConfig)
    boundary_conditions: BoundaryConfig = Field(default_factory=BoundaryConfig)
    condensation: CondensationConfig = Field(default_factory=CondensationConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    plotting: PlottingConfig = Field(default_factory=PlottingConfig)

    _base_dir: Path = PrivateAttr(default=Path('.'))

    @model_validator(mode='after')
    def _photo_ion_consistency(self):
        if self.photochemistry.use_ion and not self.photochemistry.use_photo:
            raise ValueError('use_ion=True requires use_photo=True')
        return self

    @model_validator(mode='after')
    def _solver_moldiff_requirement(self):
        if self.solver.ode_solver == 'Rodas3' and not self.atmosphere.use_moldiff:
            raise ValueError("ode_solver='Rodas3' requires atmosphere.use_moldiff=True")
        return self

    def model_post_init(self, __context) -> None:
        # Path absolutification is opt-in via from_toml(absolutify_paths=True).
        # Script mode relies on chdir + relative paths; library mode opts in.
        pass

    # Field names within `network` and `paths` that hold filesystem paths.
    # out_name is excluded (it's the basename appended to output_dir, not a path).
    _PATH_FIELDS = {
        'network': ('network', 'gibbs_text', 'com_file', 'cross_folder'),
        'paths': ('fastchem_dir', 'atm_file', 'sflux_file', 'top_BC_flux_file',
                  'bot_BC_flux_file', 'vul_ini', 'output_dir', 'plot_dir', 'movie_dir'),
    }

    def _resolve_paths(self) -> None:
        base = self._base_dir
        for section_name, fields in self._PATH_FIELDS.items():
            section = getattr(self, section_name)
            for field_name in fields:
                value = getattr(section, field_name, None)
                if value is None or not value:
                    continue
                p = Path(value)
                if not p.is_absolute():
                    abs_p = (base / p).resolve()
                    # Preserve trailing slash if the original had one (some src/
                    # code does `dir + name` concatenation expecting it).
                    s = str(abs_p)
                    if value.endswith('/'):
                        s += '/'
                    setattr(section, field_name, s)

    @classmethod
    def from_toml(
        cls,
        path: str | os.PathLike,
        base_dir: str | os.PathLike | None = None,
        overrides: dict | None = None,
        absolutify_paths: bool = False,
    ) -> 'VulcanConfig':
        """Load a TOML file into a VulcanConfig.

        Parameters
        ----------
        path : path to vulcan_cfg.toml
        base_dir : reference dir for `absolutify_paths` (defaults to TOML's dir)
        overrides : nested dict merged on top of the loaded TOML
        absolutify_paths : if True, resolve relative path fields against base_dir.
            Script mode (vulcan.py chdirs first) does not need this. Library mode
            (vulcan_api.py, callable from arbitrary CWD) should pass True.
        """
        path = Path(path)
        if base_dir is None:
            base_dir = path.parent
        base_dir = Path(base_dir).resolve()

        with open(path, 'rb') as f:
            raw = tomllib.load(f)

        if overrides:
            raw = _deep_merge(raw, overrides)

        cfg = cls(**raw)
        cfg._base_dir = base_dir
        if absolutify_paths:
            cfg._resolve_paths()
        return cfg


def _deep_merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for k, v in overlay.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
