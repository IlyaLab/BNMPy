BNMPy.experiment_data
=====================

The experiment_data module provides tools for loading and managing experimental data for PBN optimization.

.. automodule:: BNMPy.experiment_data
   :members:
   :undoc-members:
   :show-inheritance:

ExperimentData Class
--------------------

.. autoclass:: BNMPy.experiment_data.ExperimentData
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ExperimentData class loads and validates experimental data from CSV files. It handles stimuli, inhibitors, and measured values for model optimization.

Data Format
-----------

CSV file format:

.. code-block:: csv

   Experiments,Stimuli,Stimuli_efficacy,Inhibitors,Inhibitors_efficacy,Measured_nodes,Measured_values
   1,TGFa,1,TNFa,1,"NFkB,ERK,C8,Akt","0.7,0.88,0,1"
   2,TNFa,1,TGFa,1,"NFkB,ERK,C8,Akt","0.3,0.12,1,0"
   3,"TGFa,TNFa","1,1",,,"NFkB,ERK,C8,Akt","1,1,1,1"
   4,"TGFa,TNFa","1,1",PI3K,0.7,"NFkB,ERK,C8,Akt","0.3,0.12,1,0"

Column Descriptions
~~~~~~~~~~~~~~~~~~~

- **Experiments**: Experiment identifier
- **Stimuli**: Nodes fixed to 1 (comma-separated)
- **Stimuli_efficacy**: Efficacy values 0-1 (optional, defaults to 1.0)
- **Inhibitors**: Nodes fixed to 0 (comma-separated)
- **Inhibitors_efficacy**: Efficacy values 0-1 (optional, defaults to 1.0)
- **Measured_nodes**: Nodes with experimental measurements
- **Measured_values**: Corresponding values 0-1 (normalized)

Efficacy Values
~~~~~~~~~~~~~~~

- **1.0 (default)**: Full efficacy - node completely knocked out/stimulated
- **< 1.0**: Partial efficacy - creates probabilistic perturbation

  - For inhibitors (target=0): P(node=0) = efficacy, P(node=1) = 1-efficacy
  - For stimuli (target=1): P(node=1) = efficacy, P(node=0) = 1-efficacy

- **Example**: ``PI3K,0.7`` means PI3K inhibition has 70% probability of setting PI3K=0, 30% of PI3K=1

Basic Usage
-----------

.. code-block:: python

   import BNMPy

   # Load experimental data
   exp_data = BNMPy.ExperimentData("experiments.csv")

   # Access experiment information
   print(f"Number of experiments: {len(exp_data.experiments)}")
   print(f"Measured nodes: {exp_data.get_measured_nodes()}")
   print(f"Perturbed nodes: {exp_data.get_perturbed_nodes()}")

   # Iterate through experiments
   for exp in exp_data.experiments:
       print(f"Experiment {exp['id']}:")
       print(f"  Stimuli: {exp['stimuli']}")
       print(f"  Inhibitors: {exp['inhibitors']}")
       print(f"  Measurements: {exp['measurements']}")

Utility Functions
-----------------

extract_experiment_nodes
~~~~~~~~~~~~~~~~~~~~~~~~

Extract measured and perturbed nodes from experimental data:

.. code-block:: python

   measured_nodes, perturbed_nodes = BNMPy.extract_experiment_nodes("experiments.csv")
   print(f"Measured: {measured_nodes}")
   print(f"Perturbed: {perturbed_nodes}")

This is useful for model compression:

.. code-block:: python

   # Extract nodes from experiments
   measured_nodes, perturbed_nodes = BNMPy.extract_experiment_nodes("experiments.csv")

   # Compress model based on experimental information
   compressed_network, info = BNMPy.compress_model(
       network,
       measured_nodes=measured_nodes,
       perturbed_nodes=perturbed_nodes
   )

generate_experiments
~~~~~~~~~~~~~~~~~~~~

Generate hypothesized experimental values using the current PBN parameters:

.. code-block:: python

   # Generate predictions for experiments
   results_df = BNMPy.generate_experiments(
       pbn,
       experiment_csv="experiments.csv",
       output_csv="predicted_experiments.csv",
       config={'steady_state': {'method': 'monte_carlo'}}
   )

   # The output DataFrame contains:
   # - Original experiment conditions in experiment_csv
   # - Predicted values based on current PBN parameters

This is useful for:

- Validating model predictions before optimization
- Comparing different parameter sets
- Generating synthetic data for testing

See Also
--------

- :doc:`parameter_optimizer` - Main optimization interface
- :doc:`simulation_evaluator` - Simulation evaluation
- :doc:`model_compressor` - Model compression