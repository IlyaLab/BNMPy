BNMPy.model_compressor
======================

The model_compressor module provides tools for simplifying Boolean Networks before optimization.

.. automodule:: BNMPy.model_compressor
   :members:
   :undoc-members:
   :show-inheritance:


Features
--------------------

The compressor can:

1. **Remove non-observable nodes**: Nodes without paths to measured species
2. **Remove non-controllable nodes**: Nodes not influenced by perturbed species
3. **Collapse linear paths**: Simplify cascades of intermediate nodes
4. **Visualize results**: Show which nodes and edges were removed

Basic Usage
-----------

.. code-block:: python

   import BNMPy

   # Load network
   network = BNMPy.load_network("network.txt")

   # Define nodes
   measured_nodes = {'Output1', 'Output2', 'Biomarker'}
   perturbed_nodes = {'Drug1', 'Drug2', 'Input'}

   # Compress
   compressed_network, compression_info = BNMPy.compress_model(
       network,
       measured_nodes=measured_nodes,
       perturbed_nodes=perturbed_nodes,
       verbose=True
   )


Extract nodes directly from experimental data:

.. code-block:: python

   import BNMPy

   # Load network
   network = BNMPy.load_network_from_file("network.txt")

   # Extract nodes from experiments
   measured_nodes, perturbed_nodes = BNMPy.extract_experiment_nodes("experiments.csv")

   # Compress using experimental information
   compressed_network, compression_info = BNMPy.compress_model(
       network,
       measured_nodes=measured_nodes,
       perturbed_nodes=perturbed_nodes
   )


Step-by-Step Compression
------------------------

For detailed control:

.. code-block:: python

   from BNMPy.model_compressor import ModelCompressor

   # Initialize compressor
   compressor = ModelCompressor(network, measured_nodes, perturbed_nodes)

   # Analyze network
   non_observable = compressor.find_non_observable_nodes()
   non_controllable = compressor.find_non_controllable_nodes()
   collapsible_paths = compressor.find_collapsible_paths()

   print(f"Non-observable nodes: {non_observable}")
   print(f"Non-controllable nodes: {non_controllable}")
   print(f"Collapsible paths: {collapsible_paths}")

   # Selective compression
   compression_info = compressor.compress(
       remove_non_observable=True,
       remove_non_controllable=True,
       collapse_linear_paths=True
   )

   # Get compressed network
   compressed_network = compressor.get_compressed_network()

   # Get summary
   summary = compressor.get_compression_summary(compression_info)
   print(summary)

Visualization
-------------

The module provides visualization capabilities:

.. code-block:: python

   import BNMPy

   # Compress network
   compressed_network, compression_info = BNMPy.compress_model(
       network,
       measured_nodes=measured_nodes,
       perturbed_nodes=perturbed_nodes
   )

   # Visualize compression results
   BNMPy.vis_compression(
       network,                    # Original network
       compressed_network,         # Compressed network
       compression_info,           # Compression information
       "compression_results.html"  # Output file
   )

Visualization:

- **Node colors**: Perturbed nodes (red), measured nodes (orange), intermediate nodes (blue)
- **Removed nodes and edges**: Shown in light grey and dashed