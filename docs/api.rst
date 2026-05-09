API reference
=============

Simulation
----------

.. autosummary::
   :toctree: generated/

   hypersynchronization.simulate_kuramoto

Initial conditions
------------------

.. autosummary::
   :toctree: generated/

   hypersynchronization.generate_state
   hypersynchronization.generate_k_clusters
   hypersynchronization.generate_q_twisted_state

RHS functions
-------------

.. autosummary::
   :toctree: generated/

   hypersynchronization.rhs_23_sym
   hypersynchronization.rhs_pairwise_a2a
   hypersynchronization.rhs_pairwise_meso
   hypersynchronization.rhs_pairwise_adj
   hypersynchronization.rhs_pairwise_a2a_harmonic
   hypersynchronization.rhs_triplet_sym_meso
   hypersynchronization.rhs_triplet_asym_meso
   hypersynchronization.rhs_23_sym_nb
   hypersynchronization.rhs_23_asym_nb
   hypersynchronization.rhs_ring_23_sym_nb
   hypersynchronization.rhs_ring_23_asym_nb

Analysis
--------

.. autosummary::
   :toctree: generated/

   hypersynchronization.order_parameter
   hypersynchronization.identify_state
   hypersynchronization.identify_k_clusters
   hypersynchronization.identify_q_twisted

Drawing
-------

.. autosummary::
   :toctree: generated/

   hypersynchronization.plot_summary
   hypersynchronization.plot_sync
   hypersynchronization.plot_series
   hypersynchronization.plot_order_param
   hypersynchronization.plot_phases
   hypersynchronization.plot_phases_line
   hypersynchronization.plot_phases_ring
