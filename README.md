BNMPy
=====

## Installation

To install, run `pip install -e .` in this directory.

## Example usage
For a detailed example, check out the [Simple Example Notebook](./Examples/Simple_example.ipynb).

## Usage

```python
from BNMPy import booleanNetwork as bn
from BNMPy import BMatrix  

file = 'input_files/pancreatic_vundavilli_2020_fig3.txt'

network = BMatrix.load_network_from_file(file) # create a Boolean network object
ngenes = len(network.nodes)

# initial state
x0 = np.random.randint(2, size=ngenes) #random inital state 
network.setInitialValue(x0)

noise_level = 0.05 # noise
y = network.update_noise ( noise_level  , 200000) # simulation with noise
```
