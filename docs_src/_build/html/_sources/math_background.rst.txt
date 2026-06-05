Mathematical background
=======================

This chapter summarises the equations that neoVULCAN solves. The presentation
follows [Tsai2017]_ for the chemistry-only model and [Tsai2021]_
for photochemistry, condensation, advection, and the full transport flux.

Governing equation
------------------

For a one-dimensional atmosphere, the column density of each species
evolves according to the Eulerian mass-continuity equation
[Tsai2017]_, [Tsai2021]_

.. math::
   :label: eq-continuity

   \frac{\partial n_i}{\partial t}
   = \mathcal{P}_i - \mathcal{L}_i
   - \frac{\partial \phi_i}{\partial z},

where :math:`n_i\;[\mathrm{cm^{-3}}]` is the number density of species
:math:`i`, :math:`\mathcal{P}_i` and :math:`\mathcal{L}_i` are the chemical
production and loss rates :math:`[\mathrm{cm^{-3}\,s^{-1}}]`, and
:math:`\phi_i` is the vertical particle flux. The mixing ratio is
:math:`X_i = n_i / n_\mathrm{tot}` with :math:`n_\mathrm{tot} = \sum_j n_j`.

In the simplest (eddy-only) limit of [Tsai2017]_, the flux is

.. math::

   \phi_i = -\Kzz\, n_\mathrm{tot}\, \frac{\partial X_i}{\partial z}.

The full transport flux used in [Tsai2021]_ and in neoVULCAN includes
advection, eddy diffusion, molecular diffusion with hydrostatic and thermal
diffusion drifts:

.. math::
   :label: eq-flux

   \phi_i = n_i\, v
   - \Kzz\, n_\mathrm{tot}\, \frac{\partial X_i}{\partial z}
   - D_i\left[
       \frac{\partial n_i}{\partial z}
       + n_i\left(\frac{1}{H_i}
       + \frac{1+\alpha_T}{T}\,\frac{\partial T}{\partial z}\right)
     \right],

where :math:`v` is the prescribed vertical wind, :math:`D_i` the binary
molecular diffusion coefficient of species :math:`i` against the bulk gas,
:math:`H_i = k_B T / (m_i g)` the molecular scale height, and
:math:`\alpha_T` the thermal-diffusion factor.

Solving Equation :eq:`eq-continuity` for an initial condition together with
boundary conditions at the top and bottom of the atmosphere is the central
task of neoVULCAN.

Chemical production and loss
----------------------------

Two classes of reactions contribute to :math:`\mathcal{P}_i` and
:math:`\mathcal{L}_i`: thermochemical (bimolecular and termolecular) and
photochemical (photodissociation, photoionisation).

Thermochemical rate coefficients are evaluated through the modified
Arrhenius form [Tsai2017]_

.. math::

   k(T) = A\, T^{b}\, \exp\!\left(-\tfrac{E}{T}\right),

with units :math:`\mathrm{cm^3\,s^{-1}}` for bimolecular reactions and
:math:`\mathrm{cm^6\,s^{-1}}` for termolecular reactions. Three-body
reactions use the standard low-/high-pressure interpolation; the
parameters :math:`(A, b, E)` are tabulated in the network files in
``thermo/``.

Reverse rate coefficients are not fitted: they are reconstructed on the
fly from the forward rate and the equilibrium constant computed from NASA
polynomial Gibbs free energies [Tsai2017]_. This guarantees that the
chemistry is consistent with thermochemical equilibrium at every grid
point and temperature.

For each reaction :math:`R_j` of the form
:math:`\sum_\alpha \nu^R_{\alpha j}\, X_\alpha \to
\sum_\beta \nu^P_{\beta j}\, X_\beta`, the contribution to the right-hand
side of species :math:`i` is

.. math::

   (\mathcal{P}_i - \mathcal{L}_i)_j
   = (\nu^P_{ij} - \nu^R_{ij})\,
     k_j\, \prod_\alpha n_\alpha^{\nu^R_{\alpha j}}.

In neoVULCAN this expression is generated symbolically by
``make_chemistry_jax.py`` and compiled into a vectorised JAX kernel
(``chemistry_jax.chemdf``) so that it can be evaluated and differentiated
efficiently on every layer.

Photochemistry
--------------

Photodissociation is treated as a unimolecular reaction with photons
[Tsai2021]_,

.. math::

   \mathrm{A} \xrightarrow{h\nu} \mathrm{B} + \mathrm{C},

with rate coefficient

.. math::
   :label: eq-J

   k_J(z) = \int q(\lambda)\, \sigma_a(\lambda)\, J(z,\lambda)\, \d\lambda,

where :math:`\sigma_a(\lambda)` is the photoabsorption cross-section and
:math:`q(\lambda)` is the quantum yield of the specific photolysis branch.
The actinic flux :math:`J(z, \lambda)` itself is the sum of a directly
attenuated stellar beam and a diffuse component:

.. math::
   :label: eq-actinic

   J(z, \lambda) = J_\infty(\lambda)\, \exp\!\left(-\tau(z,\lambda)/\mu\right)
   + J_\mathrm{diff}(z, \lambda),

with :math:`\mu = \cos\theta` the cosine of the solar zenith angle and the
optical depth

.. math::

   \tau(z, \lambda)
   = \int_z^{z_\infty} \sum_i \bigl[\sigma_{a,i}(\lambda)
   + \sigma_{s,i}(\lambda)\bigr]\, n_i(z')\, \d z'.

The diffuse component is obtained from one of two radiative-transfer
backends, selected through ``photochemistry.rt_scheme`` in the
configuration file.

Two-stream backend (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default backend
(:class:`radiative_transfer.TwoStreamRT`) solves the delta-Eddington
two-stream equations of [Malik2019]_ and converts the resulting diffuse
flux to actinic units through the first Eddington coefficient
:math:`\epsilon =` ``photochemistry.edd`` (default 0.5):

.. math::

   J_\mathrm{diff}(z, \lambda) = F_\mathrm{diff}(z, \lambda) / \epsilon.

This is the fast, robust choice for the vast majority of exoplanet runs
and is what the original VULCAN paper [Tsai2021]_ used.

DisORT++ backend
~~~~~~~~~~~~~~~~

For configurations where the two-stream closure is too coarse — strongly
scattering atmospheres, low solar zenith angles, surface albedo studies,
or whenever a higher-fidelity reference solution is wanted — neoVULCAN
can replace the two-stream step with **DisORT++**
(:class:`radiative_transfer.DisortRT`; upstream repository:
`NewStrangeWorlds/DisORT <https://github.com/NewStrangeWorlds/DisORT>`_),
a modern C++ rewrite of the classic DISORT discrete-ordinates algorithm. Instead of a two-term
expansion of the diffuse intensity, DisORT++ discretises the angular
dependence on :math:`n_\mathrm{str}` Gauss–Legendre streams (set by
``photochemistry.disort_nstr``) and solves the resulting linear system
of radiative-transfer equations exactly per wavelength bin. Rayleigh
scattering by the bulk gases listed in ``photochemistry.scat_sp`` is
handled with the proper Rayleigh phase function rather than the
:math:`g_0` Henyey–Greenstein approximation used by the two-stream
solver, and a Lambertian lower boundary with reflectivity
``photochemistry.surface_albedo`` is supported.

For both backends, the mean intensity returned by the RT step is
converted into the actinic flux used in Equation :math:numref:`eq-J`
and passed back to the chemistry through the rate-coefficient array.
Stellar fluxes are supplied at the stellar surface in
``atm/stellar_flux/`` and rescaled to the planet by
:math:`(R_\star / a)^2`. Rayleigh scattering by bulk gases listed in
``photochemistry.scat_sp`` and temperature-dependent UV cross-sections
for the species listed in ``photochemistry.T_cross_sp`` are honoured
by both backends.

For computational efficiency, the actinic flux is recomputed every
``photochemistry.ini_update_photo_frq`` time steps while the chemistry
is still evolving rapidly and every
``photochemistry.final_update_photo_frq`` steps once it is close to
steady state.

Photoionisation
~~~~~~~~~~~~~~~

If ``use_ion = True`` the same machinery is used for photoionisation
reactions, with ionisation cross-sections taken from the same Leiden /
PHIDRATES databases that supply the photodissociation cross-sections.
Photoionisation must be run alongside photochemistry (``use_photo = True``).

Initial and elemental constraints
---------------------------------

For each elemental abundance :math:`f_X`, the particle-conservation
constraint reads [Tsai2017]_

.. math::

   \sum_i A_{X,i}\, n_i = f_X\, n_\mathrm{H},

where :math:`A_{X,i}` is the number of atoms of element :math:`X` in
species :math:`i` and :math:`n_\mathrm{H}` is the total hydrogen-atom
density. Together with the bulk-density constraint, these equations
determine consistent initial mixing ratios for any choice of
``atom_list``.

The initial condition itself is set by ``ini_mix``:

* ``'EQ'`` invokes pyFastChem to compute chemical equilibrium at the
  prescribed :math:`T`–:math:`P` profile;
* ``'const_mix'`` uses the user-supplied dictionary ``const_mix`` and
  sets all other species to zero;
* ``'vulcan_ini'`` reads a previous ``.vul`` file (the grid must match);
* ``'table'`` reads a pre-computed mixing-ratio table.


Boundary conditions
-------------------

At each of the two boundaries, neoVULCAN supports three independent
prescriptions [Tsai2021]_:

* **Flux** (``use_topflux`` / ``use_botflux``): a constant particle flux
  per unit area read from a file. Used for surface emission, surface
  deposition (typically as a deposition velocity times the local
  abundance), and for prescribed top-of-atmosphere influx.
* **Fixed mixing ratio** (``use_fix_sp_bot``): pins
  :math:`X_i = X_{i, \mathrm{bot}}` at the lower boundary. Useful when an
  ocean or surface reservoir keeps a species at a known value (e.g.,
  surface water vapour from a relative-humidity model).
* **Diffusion-limited escape** (``diff_esc``): at the top of the
  atmosphere, the upward flux of a light species is set by

  .. math::

     \phi_{i, \mathrm{top}} = -D_{i, \mathrm{top}}\, n_i \left(\frac{1}{H_i} - \frac{1}{H_0}\right),

  capped at ``max_flux`` to keep the linear system well-conditioned.

The default for species without an explicit prescription is **zero flux**,
which closes the atmosphere at that boundary. As discussed in
[Tsai2017]_, the equilibrium-chemistry lower boundary used by some
codes was found to lead to spurious vertical structure in gas giants and is
not the default in neoVULCAN.


Vertical transport
------------------

Eddy diffusion
~~~~~~~~~~~~~~

Eddy diffusion is parameterised through :math:`\Kzz`, supplied either as a
constant (``Kzz_prof = 'const'``), from a tabulated profile
(``Kzz_prof = 'file'``), or via the analytic form

.. math::

   \Kzz(P) =
   \begin{cases}
     K_\mathrm{max} & \text{if } P \ge P_\mathrm{lev}, \\[3pt]
     K_\mathrm{max}\, \bigl(P/P_\mathrm{lev}\bigr)^{-0.4}
       & \text{if } P < P_\mathrm{lev},
   \end{cases}

with :math:`K_\mathrm{max} =` ``K_max`` and :math:`P_\mathrm{lev} =`
``K_p_lev``. The eddy-diffusion flux acts on the mixing ratio gradient
:math:`\partial X_i/\partial z`, so it cannot drive a species above its
locally-uniform value: it is a smoothing term.

Molecular diffusion
~~~~~~~~~~~~~~~~~~~

The binary molecular diffusion coefficient :math:`D_i` between a minor
species and the bulk gas is computed as :math:`D_i = b_i / n_\mathrm{tot}`
with :math:`b_i` taken from the gas-kinetic tabulations of
[Tsai2021]_ (their Appendix A) for H\ :sub:`2`-, N\ :sub:`2`-, and
CO\ :sub:`2`-dominated atmospheres. The corresponding thermal-diffusion
factor :math:`\alpha_T` is read from the same table. Molecular diffusion
contributes a gravitational drift through the scale-height term
:math:`1/H_i` and a Soret drift through the
:math:`(1+\alpha_T)/T \cdot \partial T/\partial z` term, both of which can
be significant in the thermosphere of light species.

Advection
~~~~~~~~~

When ``use_vz = True`` (or ``use_vm_mol = True`` for the alternative
mixing-length parameterisation), a prescribed vertical wind :math:`v`
adds an advective component :math:`\phi^{\mathrm{adv}}_i` to the flux.
neoVULCAN discretises it with a first-order upwind scheme
[Tsai2021]_:

.. math::

   \phi^{\mathrm{adv}}_{i, j+1/2} =
   \begin{cases}
     v_{j+1/2}\, n_{i,j}     & v_{j+1/2} > 0, \\
     v_{j+1/2}\, n_{i,j+1}   & v_{j+1/2} < 0.
   \end{cases}

This is the only place in neoVULCAN where the spatial discretisation is
not centred; it is necessary because a centred discretisation of pure
advection is unconditionally unstable for the present time-stepping.


Condensation and settling
-------------------------

Condensation is implemented through schematic reactions
[Tsai2021]_

.. math::

   \mathrm{A}_\mathrm{(gas)} \leftrightarrow \mathrm{A}_\mathrm{(particle)}

whose forward (condensation) and reverse (evaporation) rates are given
by the continuum-regime growth law

.. math::
   :label: eq-cond

   \frac{\d n_\mathrm{A}}{\d t}
   = -\frac{D_\mathrm{A}\, m_\mathrm{A}}{\rho_p\, r_p^{\,2}}
     \bigl(n_\mathrm{A} - n_\mathrm{A}^\mathrm{sat}\bigr)\, n_\mathrm{A},

where :math:`D_\mathrm{A}` is the gas-phase diffusion coefficient,
:math:`m_\mathrm{A}` the molecular mass, :math:`\rho_p` and :math:`r_p`
the particle density and radius (``rho_p``, ``r_p`` in the configuration),
and :math:`n_\mathrm{A}^\mathrm{sat}` the saturation number density at the
local temperature. When :math:`n_\mathrm{A} < n_\mathrm{A}^\mathrm{sat}`
the right-hand side is positive and the term acts as an evaporation source
on the gas. Saturation curves are taken from standard expressions
(``build_atm.Atm.sp_sat``).

Condensed species sediment with Stokes' settling velocity
[Tsai2021]_

.. math::

   v_s = \frac{2}{9}\, \frac{\rho_p\, r_p^{\,2}\, g}{\mu},

where :math:`\mu` is the dynamic viscosity of the bulk gas (Cloutman
formulae). The slip-correction factor is set to unity, consistent with
the continuum-regime assumption.

Because the condensation/evaporation time-scale can be much shorter than
typical chemical time-scales, two further options are available:

* species listed in ``use_relax`` are advanced by an implicit
  relaxation step rather than by the global Rosenbrock integrator,
  which avoids time-step starvation when the system is close to
  saturation;
* after ``fix_species_time``, the species in ``fix_species`` are frozen at
  their current value (a quasi-steady-state assumption), removing them
  from the stiffness budget of the remaining chemistry.


Discretisation
--------------

The atmosphere is divided into ``nz`` layers logarithmically spaced in
pressure between ``P_b`` and ``P_t``. Number densities are layer-centred;
fluxes are interface-centred. Following [Tsai2017]_, the spatial
derivative in Equation :eq:`eq-continuity` is approximated by

.. math::

   \frac{\partial \phi_i}{\partial z}\bigg|_j
   \approx
   \frac{\phi_{i, j+1/2} - \phi_{i, j-1/2}}{\Delta z_j},

with interfacial quantities (densities, temperatures, diffusion
coefficients) defined as arithmetic averages of the two adjacent layer
values [Tsai2021]_ (their Equation 5). This staggered, finite-volume
discretisation reduces Equation :eq:`eq-continuity` to a system of ODEs
in time that can be integrated with the schemes described in
:doc:`numerics`.
