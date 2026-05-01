# ==============================================================================
# Compatibility shim — op.py re-exports everything from the split modules so
# that vulcan.py and any external code that does `import op` continues to work
# unchanged.  New code should import directly from the specific modules:
#   rates.py                — ReadRate
#   integration.py          — Integration
#   ode_solver.py           — ODESolver (spatial discretisation)
#   ros2.py                 — Ros2 (time integrator)
#   radiative_transfer.py   — TwoStreamRT, RadiativeTransfer
#   output.py               — Output
# ==============================================================================

from rates               import ReadRate
from integration         import Integration
from ode_solver          import ODESolver
from ros2                import Ros2
from radiative_transfer  import TwoStreamRT, RadiativeTransfer
from condensation        import Condensation
from output              import Output

# Re-export the shared module-level names that external code may reference
from rates import (
    species, ni, nr, nz,
    chemdf, neg_achemjac,
    compo, compo_row,
    kb, Navo, hc, ag0,
    vulcan_cfg,
    chem_funs,
    build_atm,
)
