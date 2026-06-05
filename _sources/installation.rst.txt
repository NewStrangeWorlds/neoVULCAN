Installation
============

neoVULCAN is a pure-Python code that depends on a small scientific stack
plus JAX (for the chemistry Jacobian), pyFastChem (for equilibrium
initialisation), and Pydantic (for the TOML configuration). It does
**not** require compilation: cloning the repository and installing the
Python requirements is sufficient.

Prerequisites
-------------

* **Python** 3.11 or newer (3.10 is supported via the ``tomli`` backport;
  3.11+ is preferred because ``tomllib`` is in the standard library)
* A C/C++ toolchain only if you build pyFastChem from source
  (binary wheels are available for most platforms)
* Optional: a CUDA-capable GPU and the matching ``jax[cuda12]`` wheel,
  if you want to evaluate the chemistry Jacobian on GPU
* Optional: the ``disortpp`` Python bindings for the DisORT++
  radiative-transfer backend (see :ref:`disort-install` below)


Python dependencies
-------------------

The minimum set of runtime dependencies, taken from ``requirements.txt``:

.. code-block:: text

   numpy>=2.2
   scipy>=1.15
   sympy>=1.14            # used by make_chemistry_jax.py
   matplotlib>=3.10
   Pillow>=11.0           # optional: live plotting
   jax>=0.6.2
   jaxlib>=0.6.2
   pyfastchem>=4.0        # equilibrium-chemistry initialisation
   pydantic>=2.0          # TOML schema validation
   tomli>=2.0 ;           # only required on Python 3.10

``sympy`` is only needed at code-generation time (when
``src/make_chemistry_jax.py`` runs); it is not used by the production
solver. ``Pillow`` is purely cosmetic; neoVULCAN falls back gracefully
if it is missing.


Step-by-step
------------

.. code-block:: bash

   git clone https://github.com/exoclime/VULCAN.git
   cd VULCAN/neoVULCAN
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

For a GPU build of JAX, replace the JAX lines in ``requirements.txt`` with

.. code-block:: bash

   pip install "jax[cuda12]" \
       -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

The chemistry kernel is currently configured for CPU execution in
``src/chemistry_jax.py`` (``jax.config.update('jax_platform_name', 'cpu')``);
edit that file if you want to push the chemistry onto the GPU.


.. _disort-install:

DisORT++ (optional)
-------------------

The default radiative-transfer scheme is a fast delta-Eddington
two-stream solver and ships with no extra dependencies. neoVULCAN can
also use **DisORT++** (upstream repository:
`NewStrangeWorlds/DisORT <https://github.com/NewStrangeWorlds/DisORT>`_),
a modern C++ rewrite of the classic DISORT discrete-ordinates code,
through its Python bindings ``disortpp``. Install it from PyPI

.. code-block:: bash

   pip install disortpp

(versions ≥ 2.2 are recommended because they expose
``index_from_bottom`` on ``DisortFluxConfig``, which lets neoVULCAN pass
its native bottom-to-top layer ordering through without copies). DisORT++
is *only* loaded when a run sets

.. code-block:: toml

   [photochemistry]
   rt_scheme   = "disort"
   disort_nstr = 8         # number of streams

so omitting the package is fine for two-stream runs. See
:doc:`math_background` for the physics and :doc:`numerics` for the
implementation details.


Verifying the installation
--------------------------

Run the regression tests:

.. code-block:: bash

   cd neoVULCAN
   pytest tests/

The test suite is intentionally small and fast (a few minutes on a modern
laptop) and exercises:

* the radiative-transfer modules against a stored snapshot
  (``tests/rt_snapshot.pkl``) — covers both the two-stream and the
  DisORT++ paths when ``disortpp`` is installed;
* the dormant exponential-integrator :math:`\varphi_1` function;
* an end-to-end regression on an HD 189733b-like configuration.

If the tests pass you are ready to run a science case; see
:doc:`quickstart`.
