Phenotype Score
======================

The phenotype score module provides functions to calculate phenotype scores from simulation results using gene-phenotype relationships from knowledge graph.

Overview
--------

The module integrates ProxPath pathway analysis to compute phenotype scores based on simulation results. It supports multiple input formats including:

- Genes to query (list of genes)
- Boolean Network steady state results (dictionary with fixed points and cyclic attractors)
- Probabilistic Boolean Network steady state results (numpy arrays)
- Simulation results (1D or 2D numpy arrays)
- Pandas Series/DataFrame (single or multiple rows)

Main Functions
--------------

get_phenotypes
~~~~~~~~~~~~~~

.. autofunction:: BNMPy.phenotype_score.get_phenotypes

Display available phenotypes in the ProxPath database and statistics about gene-phenotype relationships.

proxpath
~~~~~~~~

.. autofunction:: BNMPy.phenotype_score.proxpath

Extract gene-phenotype relationships from the knowledge graph for specified genes and phenotypes, return a dataframe with the relationships.

phenotype_scores
~~~~~~~~~~~~~~~~

.. autofunction:: BNMPy.phenotype_score.phenotype_scores

Calculate phenotype scores from simulation results or return formulas for manual calculation.

Usage Examples
--------------
Get Formulas Without Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
    
    # Get formulas showing how scores are calculated
    formulas = BNMPy.phenotype_scores(
        genes=['TP53', 'MYC', 'BCL2'],
        phenotypes=['APOPTOSIS', 'PROLIFERATION'],
        simulation_results=None
    )
    
    for phenotype, formula in formulas.items():
        print(f"{phenotype}: {formula}")

    # Output:
    # APOPTOSIS: TP53 + MYC + BCL2
    # PROLIFERATION: TP53 + MYC + BCL2

Basic Usage with BN Steady State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
    
    # Create network and compute steady state
    bn = BNMPy.load_network(...)
    ss_calc = BNMPy.SteadyStateCalculator(bn)
    ss_calc.set_experimental_conditions(stimuli=['TP53'])
    steady_state = ss_calc.compute_steady_state()
    
    # Calculate phenotype scores
    scores = BNMPy.phenotype_scores(
        genes=['TP53', 'MYC', 'BCL2'],
        phenotypes=['APOPTOSIS', 'PROLIFERATION'],
        simulation_results=steady_state,
        network=bn
    )
    print(scores)
    # Output:
    #                  APOPTOSIS  PROLIFERATION
    # Fixed_Point_1         0.85          -0.42
    # Fixed_Point_2         0.23           0.71
    # Cycle_1_State_1       0.45           0.12
    # Cycle_1_State_2       0.67          -0.21

With PBN Steady State
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create PBN and compute steady state
    pbn = BNMPy.load_network(...)
    ss_calc = BNMPy.SteadyStateCalculator(pbn)
    ss_calc.set_experimental_conditions(stimuli=['TP53'])
    steady_state_array = ss_calc.compute_steady_state(method='monte_carlo', n_runs=10, n_steps=1000)
    
    # Calculate phenotype scores
    scores = BNMPy.phenotype_scores(
        genes=['TP53', 'MYC', 'BCL2'],
        phenotypes=['APOPTOSIS', 'PROLIFERATION'],
        simulation_results=steady_state_array,
        network=pbn
    )
    print(scores)
    # Output:
    #                  APOPTOSIS  PROLIFERATION
    # State_1               0.85          -0.42
    # State_2               0.23           0.71
    # State_3               0.67          -0.21

With Multiple States (2D Numpy Array)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    
    # Multiple states from different conditions/time points
    # Rows = states, columns = genes
    multiple_states = np.array([
        [1, 0, 0],  # State 1
        [0, 1, 1],  # State 2
        [1, 1, 0],  # State 3
    ])
    
    scores = BNMPy.phenotype_scores(
        genes=['TP53', 'MYC', 'BCL2'],
        phenotypes=['APOPTOSIS', 'PROLIFERATION'],
        simulation_results=multiple_states,
        network=bn
    )
    print(scores)
    # Output:
    #          APOPTOSIS  PROLIFERATION
    # State_1       0.85          -0.42
    # State_2       0.23           0.71
    # State_3       0.67          -0.21

With Pandas Series
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    
    # Create simulation results as pandas Series
    sim_results = pd.Series([0.8, 0.2, 0.9], index=['TP53', 'MYC', 'BCL2'])
    
    # Calculate phenotype scores
    scores = BNMPy.phenotype_scores(
        genes=['TP53', 'MYC', 'BCL2'],
        phenotypes=['APOPTOSIS', 'PROLIFERATION'],
        simulation_results=sim_results
    )
    print(scores)
    # Output:
    #          APOPTOSIS  PROLIFERATION
    # State_1       0.85          -0.42


Gene Name Mapping
-----------------

When using simulation results, gene names can be provided in three ways (priority order):

1. **genes**: Explicitly provide a list of gene names matching the order in simulation results
2. **network**: Extract gene names from network.nodeDict
3. **simulation_results**: Extract gene names from simulation results if it is a pandas DataFrame or Series

.. code-block:: python

    # Method 1: Using genes parameter
    scores = BNMPy.phenotype_scores(
        genes=['TP53', 'MYC', 'BCL2'],
        simulation_results=steady_state_array
    )
    
    # Method 2: Using network parameter
    scores = BNMPy.phenotype_scores(
        genes=['TP53', 'MYC'],
        simulation_results=steady_state,
        network=bn  # Gene names extracted from bn.nodeDict
    )
    
    # Method 3: Using simulation_results directly
    scores = BNMPy.phenotype_scores(
        simulation_results=steady_state_array # Gene names extracted from simulation results
    )

Notes
-----

- The ProxPath database file must be available at ``BNMPy/KG_files/significant_paths_to_phenotypes.txt``
- See ``Examples/phenotype_score.ipynb`` for more examples.

References
----------

- ProxPath: https://github.com/SaccoPerfettoLab/ProxPath
- Iannuccelli, M., Vitriolo, A., Licata, L. et al. Curation of causal interactions mediated by genes associated with autism accelerates the understanding of gene-phenotype relationships underlying neurodevelopmental disorders. Mol Psychiatry 29, 186–196 (2024). https://doi-org.offcampus.lib.washington.edu/10.1038/s41380-023-02317-3
