.. BNMPy documentation master file

Welcome to BNMPy's documentation!
==================================

BNMPy is a Python library for Boolean Network (BN) and Probabilistic Boolean Network (PBN) modeling, simulation, optimization, and analysis with applications in systems biology. It also supports building models from knowledge graphs and integrating them with BNs and PBNs.
For more details, please refer to the `GitHub repository <https://github.com/ilyalab/BNMPy>`_.

Installation
------------

To install BNMPy:

.. code-block:: bash

   pip install -e .

Quick Start
-----------

.. code-block:: python

   import BNMPy

   # Load a network
   network_string = """
   A = A
   B = C
   C = !E
   D = A | B
   E = C & D
   F = !A & B
   """
   network = BNMPy.load_network_from_string(network_string, initial_state=[0, 0, 0, 0, 0, 0])

   # Visualize the network
   BNMPy.vis_network(network, output_html="SimpleBN.html", interactive=True)

   # Simulate with noise
   network.update_noise(p=0.05, iterations=10)

   # Calculate steady states
   calc = BNMPy.SteadyStateCalculator(network)
   steady_state = calc.compute_steady_state(n_runs=20, n_steps=10000)


Documentation Contents
======================

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   getting_started
   tutorials
   optimization_guide
   steady_state_guide
   compression_guide

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
