KGBN.parameter_optimizer
=========================

The parameter_optimizer module provides PBN parameter optimization based on the optPBN framework.

.. automodule:: KGBN.parameter_optimizer
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage
-----------

.. code-block:: python

   import KGBN

   # Load or create your PBN
   pbn = KGBN.load_pbn_from_file("network.txt")

   # Initialize optimizer
   optimizer = KGBN.ParameterOptimizer(
       pbn, 
       "experiments.csv", 
       nodes_to_optimize=['Cas3'],
       verbose=True,
       normalize=False  # Set to True for automatic value normalization
   )

   # Run optimization
   result = optimizer.optimize(method='differential_evolution')

   # Get optimized PBN
   if result.success:
       optimized_pbn = optimizer.get_optimized_pbn(result)

Value Normalization
-------------------

When experimental measurements span different scales, enable automatic min-max normalization:

.. code-block:: python

   optimizer = KGBN.ParameterOptimizer(
       pbn,
       "experiments.csv",
       nodes_to_optimize=['Cas3'],
       normalize=True  # Enable automatic normalization
   )

**How normalization works:**

1. **Measured values** (from CSV): All measured values across all experiments are collected and scaled to [0, 1] using min-max normalization:
   
   - Formula: ``normalized = (value - min) / (max - min)``
   - Example: Values ``[0.5, 2.0, 2.5, 3.0]`` → ``[0.0, 0.6, 0.8, 1.0]``

2. **Predicted values** (from simulation): At each optimization iteration, all predicted values (node states or formula results) across all experiments are collected and scaled to [0, 1]
   
   - Example: Predicted ``[0.1, 0.2, 0.3]`` → ``[0.0, 0.5, 1.0]``

3. **SSE calculation**: Mean squared error is computed using normalized values

Configuration
-------------

Optimization Methods
~~~~~~~~~~~~~~~~~~~~

- **Differential Evolution**: Global optimization using evolutionary strategies
- **Particle Swarm Optimization**: Swarm-based optimization

.. code-block:: python

   config = {
       # Global settings
       'seed': 9,
       'success_threshold': 0.005,
       'max_try': 3,

       # Differential Evolution parameters
       'de_params': {
           'strategy': 'best1bin',
           'maxiter': 500,
           'popsize': 15,
           'tol': 0.01,
           'mutation': (0.5, 1),
           'recombination': 0.7,
           'init': 'sobol',
           'workers': -1,
           'early_stopping': True,
       },

       # Particle Swarm Optimization parameters
       'pso_params': {
           'n_particles': 30,
           'iters': 100,
           'options': {'c1': 0.5, 'c2': 0.3, 'w': 0.9},
           'ftol': 1e-6,
           'ftol_iter': 15,
       },

       # Steady state calculation
       'steady_state': {
           'method': 'monte_carlo',
           'monte_carlo_params': {
               'n_runs': 10,
               'n_steps': 1000
           }
       }
   }

   optimizer = KGBN.ParameterOptimizer(pbn, "experiments.csv", config=config)

Discrete Mode
~~~~~~~~~~~~~

For Boolean Network optimization:

.. code-block:: python

   config = {
       'discrete_params': {
           'threshold': 0.6
       }
   }

   result = optimizer.optimize(
       method='differential_evolution',
       discrete=True
   )

Results
-------

The optimization returns an OptimizeResult object:

- ``success``: Boolean indicating successful termination
- ``message``: Status message
- ``x``: Optimized parameters
- ``fun``: Final objective value (MSE)
- ``history``: MSE values at each iteration
- ``nfev``: Number of function evaluations
- ``nit``: Number of iterations

Visualization
-------------

.. code-block:: python

   # Plot optimization history
   optimizer.plot_optimization_history(
       result, 
       save_path='optimization_history.png',
       show_stagnation=True,
       log_scale=True
   )

References
----------

Based on the optPBN framework:

Trairatphisan, P., et al. (2014). "optPBN: An Optimisation Toolbox for Probabilistic Boolean Networks." PLOS ONE 9(7): e98001.