BNMPy.model_parser
==================

The model_parser module provides functions for merging, converting, and extending Boolean and Probabilistic Boolean Networks.

.. automodule:: BNMPy.model_parser
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Merging Networks
~~~~~~~~~~~~~~~~

.. code-block:: python

   import BNMPy

   # Load two networks
   network1 = BNMPy.load_network_from_file("network1.txt")
   network2 = BNMPy.load_network_from_file("network2.txt")

   # Merge into a single Boolean Network
   merged_bn = BNMPy.merge_networks([network1, network2], output_type='BN')

   # Or merge into a Probabilistic Boolean Network
   merged_pbn = BNMPy.merge_networks([network1, network2], output_type='PBN')

Converting BN to PBN
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import BNMPy

   # Load a Boolean Network
   bn = BNMPy.load_network_from_file("network.txt")

   # Convert to PBN with equal probabilities for existing rules and a self-loop
   pbn = BNMPy.BN2PBN(bn, prob=0.5)

Extending Networks
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import BNMPy

   # Load original network and KG-derived network
   original_bn = BNMPy.load_network_from_file("original.txt")
   kg_network = BNMPy.load_signor_network(gene_list=['GENE1', 'GENE2'])

   # Extend original network with KG information
   extended_pbn = BNMPy.extend_networks(
       original_bn, 
       kg_network, 
       nodes_to_extend=['GENE1', 'GENE2'],
       prob=0.5, # probability of the rules from the KG
       descriptive=True
   )