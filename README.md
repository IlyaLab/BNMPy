BNMPy
=====

## Installation

To install, run `pip install -e .` in this directory.

## Usage

```python
from BNMPy import booleanNetwork as bn
from BNMPy import BMatrix  

file = 'input_files/pancreatic_vundavilli_2020_fig3.txt'

equations = BMatrix.get_equations(file)
gene_dict = BMatrix.get_gene_dict(equations)
upstream_genes = BMatrix.get_upstream_genes(equations)

connectivity_matrix = BMatrix.get_connectivity_matrix(equations, upstream_genes, gene_dict)
truth_table = BMatrix.get_truth_table(equations, upstream_genes)

ngenes = len(equations)

# initial state
x0 = np.random.randint(2, size=ngenes) #random inital state 
x0 = np.array(x0)

network = bn.BooleanNetwork( ngenes , connectivity_matrix, truth_table, x0  ) # create a Boolean network object
noise_level = 0.05 # noise
y = network.update_noise ( noise_level  , 200000) # simulation with noise
```
