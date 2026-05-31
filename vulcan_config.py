import os
import importlib.util


class VulcanConfig:
    """
    Per-instance configuration object that mirrors the flat vulcan_cfg module.

    Loads defaults from the vulcan_cfg.py file located in base_dir, applies
    caller-supplied overrides, and resolves all relative file paths to absolute
    paths so the library can be used from any working directory.
    """

    def __init__(self, base_dir, **overrides):
        self._base_dir = os.path.abspath(base_dir)

        # Load defaults from vulcan_cfg.py without touching sys.modules
        spec = importlib.util.spec_from_file_location(
            '_vulcan_cfg_defaults_' + str(id(self)),
            os.path.join(self._base_dir, 'vulcan_cfg.py'),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for key, val in vars(mod).items():
            if not key.startswith('_'):
                setattr(self, key, val)

        # Ensure fastchem_dir is present (may not exist in older vulcan_cfg.py files)
        if not hasattr(self, 'fastchem_dir'):
            self.fastchem_dir = 'fastchem_input'

        # Apply caller-supplied overrides after loading defaults so they win
        for k, v in overrides.items():
            setattr(self, k, v)

        # Make all relative path attributes absolute using base_dir
        self._resolve_paths()

    def _resolve_paths(self):
        path_attrs = [
            'network', 'gibbs_text', 'cross_folder', 'com_file',
            'atm_file', 'sflux_file', 'top_BC_flux_file', 'bot_BC_flux_file',
            'vul_ini', 'output_dir', 'plot_dir', 'movie_dir', 'fastchem_dir',
        ]
        for attr in path_attrs:
            val = getattr(self, attr, None)
            if val is not None and not os.path.isabs(val):
                setattr(self, attr, os.path.join(self._base_dir, val))
