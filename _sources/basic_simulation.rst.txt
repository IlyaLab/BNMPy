Basic Simulation
===============

Welcome to BNMPy! This guide will help you get started with Boolean Network and Probabilistic Boolean Network modeling.

Installation
------------

To install BNMPy, navigate to the BNMPy directory and run:

.. code-block:: bash

   cd BNMPy
   pip install -e .

Dependencies
~~~~~~~~~~~~

BNMPy requires the following packages:

- pandas
- numpy
- typing
- dataclasses
- networkx
- scipy == 1.15.2
- pyswarms == 1.3.0

These will be installed automatically when you install BNMPy using the command above.

Quick Start
-----------

Loading a Network
~~~~~~~~~~~~~~~~~

BNMPy provides a unified function to load networks from files or strings. The function automatically detects the format (Boolean Network or Probabilistic Boolean Network):

.. code-block:: python

   import BNMPy

   # From a text file
   network = BNMPy.load_network("network.txt")

   # Or from a string
   network_string = """
   A = A
   B = C
   C = !E
   D = A | B
   E = C & D
   F = !A & B
   """
   network = BNMPy.load_network(
       network_string,
       initial_state=[0, 0, 0, 0, 0, 0]
   )

Network File Format
~~~~~~~~~~~~~~~~~~~

Network files use Boolean logic syntax:

.. code-block:: text

   # Example network file
   Gene1 = Gene2 & Gene3
   Gene2 = !Gene4
   Gene3 = Gene1 | Gene5
   Gene4 = Gene4
   Gene5 = !Gene1 & Gene2

Supported operators:

- ``&`` : AND
- ``|`` : OR
- ``!`` : NOT
- ``( )`` : Grouping

Initial States
~~~~~~~~~~~~~~

You can specify initial states in two ways:

**Array format** (order matches gene order in network, can be obtained from ``network.nodeDict``):

.. code-block:: python

   network = BNMPy.load_network_from_string(
       network_string,
       initial_state=[0, 1, 0, 1, 0, 0]
   )

**Dictionary format** (explicit gene names):

.. code-block:: python

   network = BNMPy.load_network_from_string(
       network_string,
       initial_state={'A': 0, 'B': 1, 'C': 0, 'D': 1, 'E': 0, 'F': 0}
   )

Gene order can be obtained from ``network.nodeDict``.


Basic Simulation
----------------

Deterministic Update
~~~~~~~~~~~~~~~~~~~~

Synchronous update (all nodes update simultaneously):

.. code-block:: python

   # Run for 10 steps, the trajectory will be printed
   network.update(iterations=10)

   # Access current state
   print(network.nodes)


Stochastic Update
~~~~~~~~~~~~~~~~~

Add noise to represent biological uncertainty:

.. code-block:: python

   # Update with 5% flip probability
   network.update_noise(p=0.05, iterations=10)

Steady State Analysis
~~~~~~~~~~~~~~~~~~~~~

Find stable states:

.. code-block:: python

   from BNMPy import SteadyStateCalculator

   calc = SteadyStateCalculator(network)
   
   # Monte Carlo method (more accurate)
   steady_state = calc.compute_steady_state(
       method='monte_carlo',
       n_runs=20,
       n_steps=10000
   )

   # TSMC method (faster)
   steady_state = calc.compute_steady_state(
       method='tsmc',
       epsilon=0.001
   )

   print(f"Steady state probabilities: {steady_state}")

Network Visualization
---------------------

Create interactive network visualizations:

.. code-block:: python

   # Create visualization
   BNMPy.vis_network(
       network,
       output_html="network.html",
       interactive=True
   )

   # Open network.html in a browser to view


Probabilistic Boolean Networks
-------------------------------

Loading a PBN
~~~~~~~~~~~~~

PBNs are loaded using the same ``load_network()`` function. The function automatically detects the PBN format based on probabilities:

.. code-block:: python

   # From file
   pbn = BNMPy.load_network("pbn_network.txt")

   # Or from string
   pbn_string = """
   Gene1 = Gene2 & Gene3, 0.6
   Gene1 = Gene4, 0.4
   Gene2 = !Gene1
   """
   pbn = BNMPy.load_network(pbn_string)

PBN Format
~~~~~~~~~~

Each gene can have multiple rules with probabilities (format: ``gene = expression, probability``):

.. code-block:: text

   # Gene with two alternative rules
   GeneA = GeneB & GeneC, 0.7
   GeneA = !GeneD, 0.3

   # Gene with single rule (probability defaults to 1.0)
   GeneB = GeneA

Probabilities must sum to 1.0 for each gene with multiple rules.


PBN Simulation
~~~~~~~~~~~~~~

.. code-block:: python

   # Stochastic update (probabilistic rule selection)
   pbn.update_noise(p=0.01, iterations=100)

   # Calculate steady state
   calc = SteadyStateCalculator(pbn)
   steady_state = calc.compute_steady_state(
       method='monte_carlo',
       n_runs=20,
       n_steps=5000
   )

Network Manipulation
--------------------

Knockout/Knockdown
~~~~~~~~~~~~~~~~~~

Fix specific nodes to certain values:

.. code-block:: python

   # Knockout (set to 0)
   network.knockout('Gene1', value=0)

   # Stimulate (set to 1)
   network.knockout('Gene2', value=1)

   # Undo knockouts
   network.undoKnockouts()


Merging Networks
~~~~~~~~~~~~~~~~

Combine multiple networks:

.. code-block:: python

   # Load networks
   network1 = BNMPy.load_network("network1.txt")
   network2 = BNMPy.load_network("network2.txt")

   # Merge into Boolean Network using Inhibitor Wins method
   merged_bn = BNMPy.merge_networks([network1, network2], method='Inhibitor Wins')

   # Merge into PBN (creates alternative rules with probability 0.9)
   merged_pbn = BNMPy.merge_networks([network1, network2], method='PBN', prob=0.9)

More Information
----------

- :doc:`knowledge_graph_guide` - Knowledge graph integration
- :doc:`steady_state_guide` - Advanced steady state analysis
- :doc:`optimization_guide` - Parameter optimization
- :doc:`api` - Complete API reference

Examples
--------

Check the ``Examples/`` directory for Jupyter notebooks:

- ``BN_simulation.ipynb`` - Basic Boolean network simulation
- ``PBN_simulation.ipynb`` - Probabilistic Boolean network simulation
- ``BN_PBN_steady_state.ipynb`` - Steady state analysis
- ``knowledge_graph.ipynb`` - Knowledge graph integration
- ``Optimization.ipynb`` - Parameter optimization
- ``workflow_example.ipynb`` - Complete workflow
- ``BN_compression.ipynb`` - Boolean network compression

Applications
------------

See the ``pancreatic_cancer_simulation/`` directory for real-world application examples on pancreatic cancer models.
