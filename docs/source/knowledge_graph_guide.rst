Knowledge Graph Augmentation
============================

This guide covers building Boolean Networks from SIGNOR knowledge graphs, extending existing models, merging networks, and computing phenotype scores.

Overview
--------

KGBN integrates with the SIGNOR knowledge graph to:

1. **Build KG-derived BNs**: Create Boolean networks directly from SIGNOR pathways
2. **Extend existing models**: Add KG-informed rules to curated networks
3. **Merge networks**: Combine original and KG models into deterministic or probabilistic networks
4. **Phenotype scoring**: Compute phenotype scores using gene-phenotype relationships from knowledge graph

The workflow involves selecting genes of interest, fetching a Steiner subgraph from SIGNOR, converting edges to Boolean rules using joiner schemes, and optionally filtering by confidence scores.

Building KG-derived Boolean Networks
------------------------------------

Use ``KGBN.load_signor_network`` to build a Boolean network from SIGNOR:

.. code-block:: python

   import KGBN
   
   # Define genes of interest
   genes = ['KRAS', 'GNAS', 'TP53', 'SMAD4', 'CDKN2A', 'RNF43']
   
   # Build BN with different joiner schemes
   bn_string, relations = KGBN.load_signor_network(
       genes,
       joiner='inhibitor_wins',  # '&', '|', 'inhibitor_wins', 'majority', 'plurality'
       score_cutoff=0.5,        # optional: filter low-confidence edges
       only_proteins=True       # optional: restrict to protein nodes
   )
   
   # Load the network
   kg_bn = KGBN.load_network(bn_string)

Joiner Schemes
~~~~~~~~~~~~~~

The joiner parameter determines how multiple incoming edges are combined:

- **``'&'`` (AND)**: All activators AND no inhibitors
- **``'|'`` (OR)**: Any activator OR no inhibitors  
- **``'inhibitor_wins'``**: Inhibitors override activators (recommended)
- **``'majority'``**: Majority vote (tie = 0)
- **``'plurality'``**: Plurality vote (tie = 1)

Example output with inhibitor_wins:

.. code-block:: text

    CDKN2A = !MYC # Scores: MYC_inhibit:0.765
    GNAS = GNAS
    KRAS = SRC # Scores: SRC_activate:0.656
    MAPK1 = MAPK1
    MYC = !SMAD4 & MAPK1 # Scores: MAPK1_activate:0.733; SMAD4_inhibit:0.638
    SMAD4 = MAPK1 # Scores: MAPK1_activate:0.511
    SRC = GNAS # Scores: GNAS_activate:0.506
    TP53 = !SRC & MAPK1 # Scores: SRC_inhibit:0.524; MAPK1_activate:0.777

Score annotations show confidence levels for each regulation.

Extending Existing Models
-------------------------

Extend networks by adding KG-informed rules for selected nodes:

.. code-block:: python

   # Load original model
   orig_bn = KGBN.load_network('Vundavilli2020_standardized.txt')
   
   # Build KG model for same genes
   orig_genes = list(orig_bn.nodeDict.keys())
   kg_string, _ = KGBN.load_signor_network(
       orig_genes, 
       joiner='inhibitor_wins', 
       score_cutoff=0.5
   )
   kg_bn = KGBN.load_network(kg_string)
   
   # Extend specific nodes to PBN
   extended_pbn = KGBN.extend_networks(
       orig_bn, 
       kg_bn, 
       nodes_to_extend=['AKT1', 'PIK3CA'], 
       prob=0.3,  # probability for KG rules
       descriptive=True
   )

This creates a PBN where selected nodes have alternative rules from the KG with specified probabilities.

Merging Networks
----------------

Combine original and KG models using different strategies:

Deterministic Merge (Boolean Network)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Merge into deterministic BN
   merged_bn = KGBN.merge_networks(
       [orig_bn, kg_bn], 
       method='Inhibitor Wins',  # 'OR', 'AND', 'Inhibitor Wins'
       descriptive=True
   )

Probabilistic Merge (PBN)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Merge into PBN with specified probabilities
   merged_pbn = KGBN.merge_networks(
       [orig_bn, kg_bn], 
       method='PBN', 
       prob=0.9,  # probability for original model rules
       descriptive=True
   )

Merge Methods
~~~~~~~~~~~~~

- **``'OR'``**: Union of all regulations
- **``'AND'``**: Intersection of regulations  
- **``'Inhibitor Wins'``**: Inhibitors override activators (recommended)
- **``'PBN'``**: Create alternative rules with specified probabilities

Visualization and Simulation
-----------------------------

Visualize merged networks:

.. code-block:: python

   # Visualize PBN
   KGBN.vis_network(
       merged_pbn, 
       output_html="merged_network.html", 
       interactive=True
   )

Simulate the network:

.. code-block:: python

   # Calculate steady states
   calc = KGBN.SteadyStateCalculator(merged_pbn)
   steady_state = calc.compute_steady_state(n_runs=20, n_steps=10000)

Phenotype Scoring
-----------------

Compute phenotype scores using ProxPath-derived effects and simulation results:

.. code-block:: python

   # Get all available phenotypes
   KGBN.get_phenotypes()

   # Genes for phenotype scoring
   genes = list(merged_pbn.nodeDict.keys()) # or specify a list of genes

   # Get the formula for the phenotype scores without simulation results
   formulas = KGBN.phenotype_scores(
       genes=genes,
       simulation_results=None,
       phenotypes=['APOPTOSIS', 'PROLIFERATION', 'DIFFERENTIATION']
   )
   print(formulas)
   
   # Compute phenotype scores with simulation results
   scores = KGBN.phenotype_scores(
       genes=genes,
       simulation_results=steady_state,
       phenotypes=['APOPTOSIS', 'PROLIFERATION', 'DIFFERENTIATION']
   )
   
   print(scores)

The function supports multiple simulation result formats:
- pandas Series/DataFrame
- numpy arrays
- BN attractor dictionaries

References
----------
- ProxPath: https://github.com/SaccoPerfettoLab/ProxPath
- Iannuccelli, M., Vitriolo, A., Licata, L. et al. Curation of causal interactions mediated by genes associated with autism accelerates the understanding of gene-phenotype relationships underlying neurodevelopmental disorders. Mol Psychiatry 29, 186–196 (2024). https://doi-org.offcampus.lib.washington.edu/10.1038/s41380-023-02317-3


Complete Workflow Example
-------------------------

.. code-block:: python

   import KGBN
   
   # 1. Build KG-derived BN
   genes = ['KRAS', 'TP53', 'SMAD4', 'CDKN2A']
   kg_string, _ = KGBN.load_signor_network(
       genes, 
       joiner='inhibitor_wins', 
       score_cutoff=0.5
   )
   kg_bn = KGBN.load_network(kg_string)
   
   # 2. Load curated model
   orig_bn = KGBN.load_network('curated_model.txt')
   
   # 3. Merge into PBN
   merged_pbn = KGBN.merge_networks(
       [orig_bn, kg_bn], 
       method='PBN', 
       prob=0.8
   )
   
   # 4. Simulate
   calc = KGBN.SteadyStateCalculator(merged_pbn)
   steady_state = calc.compute_steady_state()
   
   # 5. Score phenotypes
   scores = KGBN.phenotype_scores(
       genes=list(merged_pbn.nodeDict.keys()),
       simulation_results=steady_state
   )
   
   # 6. Visualize
   KGBN.vis_network(merged_pbn, output_html="result.html")



See ``Examples/knowledge_graph.ipynb`` and ``Examples/phenotype_score.ipynb`` for detailed examples.