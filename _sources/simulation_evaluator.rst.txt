BNMPy.simulation_evaluator
==========================

The simulation_evaluator module provides tools for evaluating PBN simulations against experimental data.

.. automodule:: BNMPy.simulation_evaluator
   :members:
   :undoc-members:
   :show-inheritance:

SimulationEvaluator Class
-------------------------

.. autoclass:: BNMPy.simulation_evaluator.SimulationEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The SimulationEvaluator calculates the objective function for optimization by comparing PBN steady-state predictions with experimental measurements. It handles experimental conditions (stimuli/inhibitors) and computes mean squared error (MSE).

This module is automatically used by ParameterOptimizer.


See Also
--------

- :doc:`parameter_optimizer` - Main optimization interface
- :doc:`experiment_data` - Experimental data handling
- :doc:`steady_state` - Steady state calculation methods