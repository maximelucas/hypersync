API reference
=============

.. currentmodule:: hypersynchronization

Simulation
----------

.. autosummary::
   :toctree: generated/

   simulate_kuramoto

Initial conditions
------------------

.. autosummary::
   :toctree: generated/

   generate_state
   generate_k_clusters
   generate_q_twisted_state

RHS functions
-------------

.. autosummary::
   :toctree: generated/

   rhs_23_sym
   rhs_pairwise_a2a
   rhs_pairwise_meso
   rhs_pairwise_adj
   rhs_pairwise_a2a_harmonic
   rhs_triplet_sym_meso
   rhs_triplet_asym_meso
   rhs_23_sym_nb
   rhs_23_asym_nb
   rhs_ring_23_sym_nb
   rhs_ring_23_asym_nb

Analysis
--------

.. autosummary::
   :toctree: generated/

   order_parameter
   identify_state
   identify_k_clusters
   identify_q_twisted

Drawing
-------

.. autosummary::
   :toctree: generated/

   plot_summary
   plot_sync
   plot_series
   plot_order_param
   plot_phases
   plot_phases_line
   plot_phases_ring
