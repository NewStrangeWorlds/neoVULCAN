Numerical methods
=================

After spatial discretisation, neoVULCAN solves a system of
:math:`N_i \cdot N_z` ordinary differential equations of the form

.. math::

   \frac{\d \mathbf{n}}{\d t} = \mathbf{f}(\mathbf{n})
   = \mathbf{f}_\mathrm{chem}(\mathbf{n})
   + \mathbf{f}_\mathrm{tran}(\mathbf{n}),

where :math:`\mathbf{f}_\mathrm{chem}` collects the chemical production
and loss and :math:`\mathbf{f}_\mathrm{tran}` collects the divergence of
the transport flux from :doc:`math_background`. This system is **very
stiff**: in a typical C–H–N–O–S network the ratio of the fastest to the
slowest eigenvalue of the Jacobian is between :math:`10^{20}` and
:math:`10^{30}` [Tsai2017]_. Implicit or semi-implicit time stepping
is therefore unavoidable.


Why Rosenbrock methods
----------------------

Fully implicit methods (e.g. backward Euler with Newton iteration) handle
the stiffness but require an inner nonlinear solve at every step, which
is expensive when each step has :math:`\sim 10^4` unknowns. Rosenbrock
W-methods avoid the inner Newton loop by linearising the implicit
relation around the current solution [Tsai2017]_,

.. math::

   \mathbf{n}_{k+1}
   = \mathbf{n}_k + \Delta t\, (\mathbf{I} - \Delta t\, J)^{-1}\,
     \mathbf{f}(\mathbf{n}_k),

with :math:`J = \partial \mathbf{f}/\partial \mathbf{n}` the Jacobian of
the right-hand side. Each step costs a single (banded) linear solve.
Verwer et al. (1997) recommended the second-order Rosenbrock method for
chemical kinetics for being stable over large step sizes while requiring
only a Jacobian evaluation and a banded linear solve per step. That
recommendation, originally adopted in [Tsai2017]_, still drives the
default ``Ros2`` solver in neoVULCAN.


Ros2: second-order Rosenbrock
-----------------------------

The two-stage Rosenbrock method used by neoVULCAN follows Verwer et al.
(1997). With :math:`\gamma = 1 + 1/\sqrt{2}` the stages are

.. math::

   (\mathbf{I} - \gamma\Delta t\, J)\, \mathbf{g}_1
     &= \mathbf{f}(\mathbf{n}_k), \\
   (\mathbf{I} - \gamma\Delta t\, J)\, \mathbf{g}_2
     &= \mathbf{f}(\mathbf{n}_k + \Delta t\, \mathbf{g}_1)
        - 2\,\mathbf{g}_1, \\
   \mathbf{n}_{k+1} &= \mathbf{n}_k
     + \tfrac{3}{2}\Delta t\, \mathbf{g}_1
     + \tfrac{1}{2}\Delta t\, \mathbf{g}_2.

The matrix :math:`(\mathbf{I} - \gamma\Delta t\, J)` is the same in both
stages, so it is LU-factorised once with ``dgbtrf`` and back-substituted
twice with ``dgbtrs``. This is the dominant cost of a step and the main
reason Ros2 is efficient.

The embedded first-order estimate
:math:`\mathbf{n}^{*}_{k+1} = \mathbf{n}_k + \Delta t\, \mathbf{g}_1`
gives the local truncation error :math:`\mathcal{E} = |\mathbf{n}_{k+1} -
\mathbf{n}^{*}_{k+1}| = \mathcal{O}(\Delta t^{2})`, which is used to adapt
the step size [Tsai2017]_,

.. math::

   \Delta t_{k+1} = 0.9\, \Delta t_k\, \bigl(\mathcal{T}/\mathcal{E}\bigr)^{0.5},

where :math:`\mathcal{T} =` ``rtol`` is the user-requested relative
tolerance and the prefactor 0.9 is a safety factor. ``dt_var_max`` and
``dt_var_min`` bound the per-step growth and shrink ratios; ``dt_min``
and ``dt_max`` are hard limits.

A step is **rejected** when any of the following occur:

* the estimated truncation error exceeds :math:`\mathcal{T}`;
* any updated number density would be more negative than ``nega_cut``;
* element conservation would drift by more than ``loss_eps``.

Rejected steps re-use the same Jacobian and reduce :math:`\Delta t` by
``dt_var_min``.

Ros2 is A-stable: every eigenmode of the linearised system decays for any
:math:`\Delta t`, so the integrator never goes unstable, only inaccurate.


Rodas3: third-order, L-stable Rosenbrock–Wanner
-----------------------------------------------

For configurations where Ros2's second-order accuracy forces small steps
even when the chemistry is nearly relaxed, neoVULCAN also implements the
``Rodas3`` scheme (Sandu, Verwer, Van Loon, Carmichael, Potra, Dabdub &
Seinfeld 1997; KPP canonical form). It is a four-stage,
third-order-accurate, **L-stable, stiffly-accurate** Rosenbrock–Wanner
method. With :math:`\gamma = 1/2` and the W-matrix
:math:`W = \mathbf{I}/(\gamma\Delta t) - J` the stages are

.. math::

   W\, \mathbf{k}_1 &= \mathbf{f}(\mathbf{n}_k), \\
   W\, \mathbf{k}_2 &= \mathbf{f}(\mathbf{n}_k)
                       + \tfrac{4}{\Delta t}\, \mathbf{k}_1, \\
   W\, \mathbf{k}_3 &= \mathbf{f}(\mathbf{n}_k)
                       + \tfrac{1}{\Delta t}\, \mathbf{k}_1
                       - \tfrac{1}{\Delta t}\, \mathbf{k}_2, \\
   W\, \mathbf{k}_4 &= \mathbf{f}(\mathbf{n}_k + 2\mathbf{k}_1 + \mathbf{k}_3)
                       + \tfrac{1}{\Delta t}\, \mathbf{k}_1
                       - \tfrac{1}{\Delta t}\, \mathbf{k}_2
                       - \tfrac{8}{3\Delta t}\, \mathbf{k}_3, \\
   \mathbf{n}_{k+1} &= \mathbf{n}_k + 2\mathbf{k}_1 + \mathbf{k}_3 + \mathbf{k}_4.

The third stage re-uses :math:`\mathbf{f}` from the second, so only three
RHS evaluations and four banded back-substitutions per step are required.
The embedded estimate is simply :math:`|\mathbf{k}_4|`, which gives a
second-order error indicator.

Rodas3 is more expensive per step than Ros2 (roughly :math:`2\times`),
but its higher order and L-stability mean that it can take larger steps
when the chemistry stops changing rapidly. In practice it becomes
favourable when very tight convergence is required.

In the current implementation Rodas3 supports only the standard
configuration (``use_moldiff = True``, ``use_settling = False``,
``use_vm_mol = False``); use Ros2 for the other paths.


The Jacobian
------------

The Jacobian :math:`J = \partial \mathbf{f}/\partial \mathbf{n}` is the
sum of a chemistry part and a transport part. Because diffusion only
couples a layer to its two neighbours, :math:`J` has a **block
tridiagonal** structure with one :math:`N_i \times N_i` block per layer
and two off-diagonal blocks of the same size [Tsai2017]_ (their
Figure 14). neoVULCAN stores this matrix in LAPACK banded format and uses
``dgbtrf``/``dgbtrs`` to factorise and solve it.

Chemistry block
~~~~~~~~~~~~~~~

The chemistry block is computed exactly by JAX's forward-mode automatic
differentiation, applied to ``chemistry_jax.chemdf``. Concretely,

.. code-block:: python

   chem_jac_blocks = jax.jacfwd(_chemdf_single)
   # shape: (nz, ni, ni), one dense block per layer

is ``vmap``-ed over layers and JIT-compiled. The resulting kernel is
called once per Rosenbrock step and reused for all stages of that step.
Because the auto-generated ``chemdf`` is symbolic in the rate
coefficients, ``jacfwd`` gives the analytical Jacobian to machine
precision and is much cheaper than re-running the symbolic differentiation
of the original VULCAN.

Transport blocks
~~~~~~~~~~~~~~~~

Eddy and molecular diffusion contribute a constant (in :math:`\mathbf{n}`)
tridiagonal block per species. These are assembled in
``jacobian_jax._lhs_jac_banded_kernel``, which also adds the
:math:`c_0\,\mathbf{I}` term required by the Rosenbrock W-matrix and the
boundary-condition rows. The resulting banded matrix is returned in the
LAPACK layout ``(2\,b_w + 1) \times (N_i N_z)`` ready for ``dgbtrf``.

Cost
~~~~

The factorisation is :math:`\mathcal{O}(N_z\, N_i^{\,2})` and the
back-substitution is :math:`\mathcal{O}(N_z\, N_i)`. The Jacobian
assembly is dominated by the JAX chemistry block; for a network with
:math:`\sim 100` species and ``nz = 120`` it takes a few milliseconds per
step on a modern CPU.


Optional damped-Newton finisher
-------------------------------

When ``solver.use_newton_finisher = true`` the integrator switches from
the Rosenbrock step to a short **damped-Newton tail** once the per-step
fractional change drops below ``solver.newton_switch_dy``. The aim is
not to gain extra orders of accuracy but to remove the residual that a
linearly-implicit method leaves at the steady state without spending
the further Rosenbrock steps that would otherwise be needed to
extinguish it. Each Newton iteration

#. evaluates :math:`\mathbf{f}(\mathbf{n})` and
   :math:`J = \partial \mathbf{f}/\partial \mathbf{n}`;
#. solves the banded linear system
   :math:`J\, \Delta \mathbf{n} = -\mathbf{f}(\mathbf{n})`;
#. line-searches the damping factor :math:`\alpha \in
   [\text{\texttt{newton\_alpha\_min}}, 1]` so that
   :math:`\mathbf{n} + \alpha\, \Delta \mathbf{n}` stays positive and
   the residual decreases;

and the finisher stops as soon as the residual norm is below
``solver.newton_res_tol`` or after ``solver.newton_max_iter`` iterations,
whichever comes first. After it has fired, the Rosenbrock loop is
re-entered with a cool-down of ``solver.newton_cooldown`` steps before
Newton can fire again. This is an opt-in convenience for very flat
steady states; it is off by default and can be ignored for typical
runs.


Radiative-transfer backends
---------------------------

Photochemistry needs an actinic flux on every grid layer; how that flux
is obtained from the layer optical depth, single-scattering albedo, and
phase function depends on the value of ``photochemistry.rt_scheme``.

Two-stream backend
~~~~~~~~~~~~~~~~~~

:class:`radiative_transfer.TwoStreamRT` implements the standard
delta-Eddington two-stream method of [Malik2019]_ in pure NumPy. Per
wavelength bin it propagates the direct stellar beam analytically,
solves a tridiagonal system for the diffuse up- and down-going fluxes,
and converts the resulting mean intensity to actinic flux through the
first Eddington coefficient ``photochemistry.edd``. The stage costs
:math:`\mathcal{O}(N_z N_\lambda)` and is fully vectorised over the
wavelength axis; for typical hot-Jupiter and terrestrial set-ups it is
the cheapest part of a time step. A single asymmetry factor
``ag0`` (defined in :mod:`phy_const`, default ``0``) is used for all
scatterers; switch to DisORT++ if a wavelength-dependent or
species-resolved phase function is required.

DisORT++ backend
~~~~~~~~~~~~~~~~

:class:`radiative_transfer.DisortRT` replaces the two-stream flux step
with a per-bin call to the C++ **DisORT++** discrete-ordinates solver.
DisortRT subclasses TwoStreamRT and overrides only the flux step, so the
optical-depth assembly (``_compute_tau``) and the J / J_ion spectral
integrals (``_compute_J``, ``_compute_Jion``) are inherited unchanged.
At construction time the class

* imports the optional ``disortpp`` package (raising at construction
  time, not at first use, so misconfiguration surfaces early);
* creates a single ``DisortFluxConfig`` of size :math:`N_z` with
  ``photochemistry.disort_nstr`` streams (and the Rayleigh phase
  function);
* detects DisORT++ ≥ 2.2 via the presence of ``index_from_bottom`` on
  the config object and switches off the layer-axis reversals that
  earlier versions need (neoVULCAN's native ordering is bottom-to-top).

On every call the per-layer optical depth and single-scattering albedo
are reconstructed from the cumulative optical depth assembled in
``_compute_tau``, the surface albedo is refreshed from
``photochemistry.surface_albedo``, and a **single batch call**
``solve_flux_spectral`` is issued. DisORT++ then loops over all
wavelength bins inside C++ with OpenMP, so the per-step Python overhead
remains constant in the number of bins; the dominant cost scales as
:math:`\mathcal{O}(N_z\, n_\mathrm{str}^{\,3}\, N_\lambda)`. The returned
mean intensity is converted to actinic flux as
:math:`J = 4\pi\, \bar I` and stored in the same ``var.aflux`` array
that the two-stream backend uses, so the rest of the convergence and
plotting pipeline is unchanged.

When to choose which
~~~~~~~~~~~~~~~~~~~~

The two-stream backend is the right default: it is cheap, robust, and
has been validated against the reference simulations of [Tsai2021]_.
DisORT++ is preferable when

* the atmosphere is strongly scattering and the Eddington closure
  breaks down (high-albedo cases, deep Rayleigh layers),
* the solar zenith angle is low (small :math:`\mu`),
* a non-zero surface albedo is needed,
* a higher-order angular discretisation is wanted as a reference for
  validating two-stream results.

The DisORT++ backend is several times more expensive per call but is
called only every ``ini_update_photo_frq`` / ``final_update_photo_frq``
chemistry steps, so the wall-clock impact on a converged run is usually
modest.


Step-size control
-----------------

In addition to the truncation-error controller above, neoVULCAN
optionally adjusts the **relative tolerance** ``rtol`` itself during a
run (``use_adapt_rtol``). The motivation is element conservation: if the
running total of any atomic element drifts by more than ``loss_eps``
between two convergence checks, ``rtol`` is reduced (within
``[rtol_min, rtol_max]``) and the step is retried. After condensation has
reached quasi-equilibrium (i.e. after ``fix_species_time``), the solver
switches to the tighter ``post_conden_rtol`` to lock in the cloud
distribution.

A simple proportional–integral (PI) controller is available as an opt-in
alternative to the default integrating controller; see
``tests/bench_pi_controller.py`` for the benchmarks. The PI controller
sometimes improves performance on extremely stiff sulphur networks but
has not become the default after the experiments documented in
``tests/integrator_attempts_history.md``.


Steady-state criterion
----------------------

The integration is terminated when either of two compound conditions is
satisfied [Tsai2017]_:

.. math::

   \max_i \Delta \hat n_i < \delta
   \quad \text{and} \quad
   \max_i \frac{\Delta \hat n_i}{\Delta \tau} < \epsilon,

with

.. math::

   \Delta \hat n_i \equiv \frac{|n_{i, k} - n_{i, k'}|}{n_{i, k}},
   \qquad
   \Delta \tau \equiv t_k - t_{k'},

where :math:`k'` is the step at fraction ``st_factor`` of the current
integration time (default :math:`f = 0.5`, so the last half of the run is
examined). In neoVULCAN, :math:`\delta =` ``yconv_cri`` (default
:math:`10^{-2}`) and :math:`\epsilon =` ``slope_cri`` (default
:math:`10^{-4}\,\mathrm{s^{-1}}`). A relaxed alternative
(``yconv_min``, ``slope_min``) is also accepted, and when photochemistry
is active the additional condition

.. math::

   |\Delta J / J| < \text{\texttt{flux\_cri}}

must also hold to ensure the actinic flux itself has converged.

Hard limits
~~~~~~~~~~~

If steady state is not reached within ``count_max`` steps or wall-clock
time ``runtime``, the run terminates with ``end_case != 1``. The result
is still pickled, but the user should treat it as a snapshot, not as a
converged solution.


Discretisation summary
----------------------

Putting the pieces together, one neoVULCAN time step computes

#. the chemistry RHS through ``chemistry_jax.chemdf``;
#. the transport RHS through ``ode_solver.ODESolver.diffdf`` (or a
   variant for advection/settling);
#. the Jacobian through ``jacobian_jax._lhs_jac_banded_kernel``;
#. an LU factorisation of the W-matrix with ``dgbtrf``;
#. the Rosenbrock stages with the chosen scheme (``ros2`` or ``rodas3``);
#. the truncation-error check, step rejection, and step-size update;
#. periodically (every ``ini_update_photo_frq`` or
   ``final_update_photo_frq`` steps) a radiative-transfer update.

After each accepted step the integration time is advanced, the
convergence and element-conservation diagnostics are recorded, and the
loop continues until one of the termination criteria above fires.
