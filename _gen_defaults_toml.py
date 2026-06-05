"""Regenerate vulcan_cfg_defaults.toml from the schema.

Run from the neoVULCAN/ directory:
    python _gen_defaults_toml.py

Update this script whenever derived/aliased fields change in the schema so the
documentation stays in sync.
"""

import types
from typing import Literal, get_args, get_origin

from neovulcan_config import VulcanConfig


DERIVED = {
    ('atmosphere', 'para_anaTP'): 'aliased to atmosphere.para_warm when omitted',
    ('condensation', 'fix_species_time'): 'aliased to condensation.stop_conden_time when omitted',
    ('solver', 'dt_max'): 'derived as solver.runtime * 1e-5 when omitted',
    ('plotting', 'save_movie_rate'): 'aliased to plotting.live_plot_frq when omitted',
}

PROPERTIES = {
    'atmosphere': [('sl_angle', 'radians = math.radians(sl_angle_deg) — read-only')],
}


def annot_str(annotation):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return ' | '.join(repr(a) for a in args)
    if args and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return f'optional {annot_str(non_none[0])}'
    if origin is list:
        return f'list[{annot_str(args[0])}]'
    if origin is dict:
        return f'dict[{annot_str(args[0])}, {annot_str(args[1])}]'
    return getattr(annotation, '__name__', str(annotation))


def toml_repr(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return '[' + ', '.join(toml_repr(v) for v in value) + ']'
    if isinstance(value, dict):
        if not value:
            return '{}'
        return '{ ' + ', '.join(f'{k} = {toml_repr(v)}' for k, v in value.items()) + ' }'
    if isinstance(value, float):
        if value == 0:
            return '0.0'
        ax = abs(value)
        if ax >= 1e5 or ax < 1e-3:
            s = f'{value:g}'.replace('e+0', 'e').replace('e+', 'e').replace('e-0', 'e-')
            return s if 'e' in s else f'{value:.1e}'
        return repr(value)
    return repr(value)


def generate():
    lines = [
        '# neoVULCAN configuration: schema defaults reference',
        '#',
        '# Generated from neovulcan_config.VulcanConfig by _gen_defaults_toml.py.',
        '# Every field is shown with its default value (used when the TOML omits it).',
        '# Required fields are marked "# REQUIRED" and shown with a placeholder.',
        '#',
        '# This file is for DOCUMENTATION ONLY — copy a section into your own TOML',
        '# and override what you need. To regenerate after a schema change:',
        '#   python _gen_defaults_toml.py',
        '',
    ]
    for sec_name, sec_info in VulcanConfig.model_fields.items():
        sec_cls = sec_info.annotation
        if not hasattr(sec_cls, 'model_fields'):
            continue
        lines.append(f'[{sec_name}]')
        for fname, finfo in sec_cls.model_fields.items():
            type_str = annot_str(finfo.annotation)
            bits = [type_str]
            if (sec_name, fname) in DERIVED:
                bits.append(DERIVED[(sec_name, fname)])
            comment = '  # ' + ' — '.join(bits)

            if finfo.is_required():
                if 'list' in type_str:
                    placeholder = '[]'
                elif 'dict' in type_str:
                    placeholder = '{}'
                elif type_str in ('float', 'int'):
                    placeholder = '0.0'
                else:
                    placeholder = '"..."'
                lines.append(f'# REQUIRED:  {fname} = {placeholder}{comment}')
            else:
                try:
                    default = finfo.get_default(call_default_factory=True)
                except Exception:
                    default = finfo.default
                if default is None:
                    lines.append(f'# (omitted by default) {fname}{comment}')
                elif isinstance(default, dict) and any(isinstance(v, dict) for v in default.values()):
                    lines.append(f'# {fname} = {{ ... }}{comment}')
                else:
                    lines.append(f'{fname} = {toml_repr(default)}{comment}')
        for prop_name, prop_doc in PROPERTIES.get(sec_name, []):
            lines.append(f'# (derived property) {prop_name}  # {prop_doc}')
        lines.append('')

    return '\n'.join(lines)


if __name__ == '__main__':
    content = generate()
    with open('vulcan_cfg_defaults.toml', 'w') as f:
        f.write(content)
    print(f'Wrote vulcan_cfg_defaults.toml ({content.count(chr(10))} lines)')
