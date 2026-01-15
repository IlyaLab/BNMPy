BNMPy.experiment_data
=====================

The ExperimentData class loads and validates experimental data from CSV files. It handles stimuli, inhibitors, and measured values for model optimization.

.. automodule:: BNMPy.experiment_data
   :members:
   :undoc-members:
   :show-inheritance:


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
- **Measured_nodes**: Nodes with experimental measurements OR a formula expression (when use_formula=True)
- **Measured_values**: Corresponding values 0-1 (normalized). For formulas, provide a single value per row in the formula's natural range

Efficacy Values
~~~~~~~~~~~~~~~

- **1.0 (default)**: Full efficacy - node completely knocked out/stimulated
- **< 1.0**: Partial efficacy - creates probabilistic perturbation

  - For inhibitors (target=0): P(node=0) = efficacy, P(node=1) = 1-efficacy
  - For stimuli (target=1): P(node=1) = efficacy, P(node=0) = 1-efficacy

- **Example**: ``PI3K,0.7`` means PI3K inhibition has 70% probability of setting PI3K=0, 30% of PI3K=1

Formula-based Measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using formulas (``use_formula=True``):

- **Measured_nodes**: Contains a formula expression (e.g., ``N1 + N2 - N3``)
- **Measured_values**: Single value per row in the formula's natural range
- Supported operators: ``+``, ``-``, ``*``, ``/``, parentheses
- Variables must be node names from the network

**Important**: Ensure measured values are scaled to match the theoretical formula range:

- ``N1 + N2 + N3``: range [0, 3]
- ``N1 + N2 - N3``: range [-1, 2]  
- ``N1 - N2``: range [-1, 1]

The optimizer will warn if measured values fall outside the expected range.
