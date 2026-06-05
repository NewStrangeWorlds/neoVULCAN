"""Process-wide singleton holding the loaded VulcanConfig instance."""

from __future__ import annotations

from neovulcan_config import VulcanConfig

_CFG: VulcanConfig | None = None


def get_cfg() -> VulcanConfig:
    if _CFG is None:
        raise RuntimeError(
            'Config not loaded. Call set_cfg() from an entry point '
            '(vulcan.py, vulcan_api.py) before importing src/ modules.'
        )
    return _CFG


def set_cfg(cfg: VulcanConfig) -> None:
    global _CFG
    _CFG = cfg


def clear_cfg() -> None:
    global _CFG
    _CFG = None


def get_cfg_or_load(toml_path: str = 'vulcan_cfg.toml',
                    base_dir: str | None = None) -> VulcanConfig:
    """Return the singleton config, loading it from TOML if not yet set.

    Convenience for standalone scripts (plot_py/*, atm/* helpers) that aren't
    launched through vulcan.py and so need to set up the singleton themselves.
    Pass `base_dir` if relative paths in the TOML should be resolved against a
    directory other than the TOML's own location.
    """
    if _CFG is None:
        import os as _os
        from neovulcan_config import VulcanConfig as _VC
        if base_dir is None:
            base_dir = _os.path.dirname(_os.path.abspath(toml_path))
        set_cfg(_VC.from_toml(toml_path, base_dir=base_dir))
    return _CFG
