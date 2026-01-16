# KGBN: Knowledge Graph extended Boolean Network modeling

A Python library for Boolean Network (BN) and Probabilistic Boolean Network (PBN) modeling, extension, optimization, and analysis with applications in systems biology.

See [https://ilyalab.github.io/KGBN/](https://ilyalab.github.io/KGBN/) for detailed documentation.

## Installation

To install, run `pip install -e .` in this directory.

## Core functionalities

### 1. Boolean network

#### Basic operations

- **Network loading**: Load networks from text files, SBML files, or string representations
- **Network construction**: Build networks from connectivity matrices and Boolean functions
- **Network manipulation**: Knockout/knockdown specific nodes, rename nodes, merge networks
- **Network visualization**: Interactive network visualizations with igraph and matplotlib

#### Simulation

- **Deterministic update**: Synchronous and asynchronous update schemes
- **Stochastic update**: Add noise to represent biological uncertainty
- **Attractor analysis**: Find steady states and limit cycles

### 2. Probabilistic Boolean network

Basic operations are similar to BNs.

- **Stochastic Simulation**: Run Monte Carlo simulations over steps
- **Attractor analysis**: Find steady states (state distributions) via Monte Carlo or Two-state Markov Chain (TSMC) methods

### 3. Knowledge Graph (KG) augmentation

KGBN can build and extend models using the SIGNOR knowledge graph.

* **Build KG-derived BN**: Build a Boolean Network (BN) directly from a Knowledge Graph (KG)
* **Merge models**: Merge an original BN with a KG-derived model into either a BN or a PBN
* **Targeted extension to PBN**: Extend an existing BN by adding nodes and rules informed by KG
* **Phenotype scoring from KG paths**: Compute phenotype scores using gene-phenotype relations from KG

### 4. PBN Optimization

#### Parameter optimization

- **Objective function**: MSE-based optimization for experimental data fitting
- **Optimization algorithms**: Differential evolution, particle swarm optimization
- **Discrete optimization**: Support for discrete parameter spaces
- **Result Evaluation**: Compare predictions with experimental observations
- **Model compression**: Simplify models while preserving behavior

## Examples

Tutorials:

- **[BN_simulation.ipynb](./Examples/BN_simulation.ipynb)**: Basic Boolean network simulation
- **[PBN_simulation.ipynb](./Examples/PBN_simulation.ipynb)**: Probabilistic Boolean network simulation
- **[knowledge_graph.ipynb](./Examples/knowledge_graph.ipynb)**: Build from SIGNOR, extend, and merge into BN/PBN
- **[phenotype_score.ipynb](./Examples/phenotype_score.ipynb)**: Calculate phenotype scores using simulation results
- **[Optimization.ipynb](./Examples/Optimization.ipynb)**: Parameter optimization with experimental data
- **[workflow_example.ipynb](./Examples/workflow_example.ipynb)**: Complete workflow from data to optimized model

Applications:

* **[Optimization_Eduati2020_reducedData.ipynb](./pancreatic_cancer_simulation/Optimization_Eduati2020_reducedData.ipynb)**: Application on a pancreatic cancer model (Eduati 2020)
* AML example

Other utilities:

- **[BN_PBN_steady_state.ipynb](./Examples/BN_PBN_steady_state.ipynb)**: Steady state analysis
- **[BN_compression.ipynb](./Examples/BN_compression.ipynb)**: Model compression and simplification

## Quick Start

```python
sys.path.append('./src')
import KGBN

# Load a network (auto-detects BN vs PBN, file vs string)
network_string = """
A = A
B = C
C = !E
D = A | B
E = C & D
F = !A & B
"""
# initial state (array or dictionary)
x0 = [0, 0, 0, 0, 0, 0]  # array: order matches gene order in string
# or
x0 = {'A': 0, 'B': 1, 'C': 0, 'D': 0, 'E': 0, 'F': 0}  # dict: keys are gene names
network = KGBN.load_network(network_string, initial_state=x0)

# Visualize the network
KGBN.vis_network(network, output_html="SimpleBN.html", interactive=True)

# Simulate with noise = 0.05 for 10 steps
network.update_noise(p=0.05, iterations=10)

# Calculate steady states
calc = KGBN.SteadyStateCalculator(network)
steady_state = calc.compute_steady_state(n_runs=20,n_steps=10000)

```

Note that when assigning initial states or obtaining steady states, gene order matches their order in the network string/file, and can be obtained by calling `network.nodeDict`. Currently it support both array input and dictionary input (genes not specified default to 0.)

## Documentation

- **Functions**: [https://ilyalab.github.io/KGBN/](https://ilyalab.github.io/KGBN/) (in development)
- **Tutorials**: Check the [Examples](./Examples/) directory for tutorials
- **Optimization Guide**: See [Optimizer README](./src/Optimizer_README.md) for advanced optimization features
