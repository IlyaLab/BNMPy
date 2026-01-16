KGBN.model_parser
==================

The model_parser module provides functions for merging, converting, and extending Boolean and Probabilistic Boolean Networks.

.. automodule:: KGBN.model_parser
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Merging Networks
~~~~~~~~~~~~~~~~

.. code-block:: python

   import KGBN

   # Load two networks
   network1 = KGBN.load_network_from_file("network1.txt")
   network2 = KGBN.load_network_from_file("network2.txt")

   # Merge into a single Boolean Network
   merged_bn = KGBN.merge_networks([network1, network2], output_type='BN')

   # Or merge into a Probabilistic Boolean Network
   merged_pbn = KGBN.merge_networks([network1, network2], output_type='PBN')

Converting BN to PBN
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import KGBN

   # Load a Boolean Network
   bn = KGBN.load_network_from_file("network.txt")

   # Convert to PBN with equal probabilities for existing rules and a self-loop
   pbn = KGBN.BN2PBN(bn, prob=0.5)

Extending Networks
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import KGBN

   # Load original network and KG-derived network
   original_bn = KGBN.load_network_from_file("original.txt")
   kg_network = KGBN.load_signor_network(gene_list=['GENE1', 'GENE2'])

   # Extend original network with KG information
   extended_pbn = KGBN.extend_networks(
       original_bn, 
       kg_network, 
       nodes_to_extend=['GENE1', 'GENE2'],
       prob=0.5, # probability of the rules from the KG
       descriptive=True
   )