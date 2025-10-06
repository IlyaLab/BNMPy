# BNMPy: Boolean network modeling and optimization

A Python library for Boolean Network (BN) and Probabilistic Boolean Network (PBN) modeling, simulation, optimization, and analysis with applications in systems biology.

## Installation

To install, run `pip install -e .` in this directory.

## Core functionalities

### 1. Boolean network simulation

#### Basic operations

- **Network loading**: Load networks from text files, SBML files, or string representations
  - `load_network_from_file`, `load_network_from_string`
- **Network construction**: Build networks from connectivity matrices and Boolean functions
  - See [BMatrix README](.src/BMatrix_README.md)
- **Network manipulation**: Knockout/knockdown specific nodes, rename nodes, merge networks
  - `merge_networks`, `BN2PBN`, `extend_networks`
- **Network visualization**: Interactive network visualizations with igraph and matplotlib
  - `vis_network`, `vis_compression`, `vis_extension`

#### Simulation

- **Deterministic update**: Synchronous and asynchronous update schemes
  - `network.update()`
- **Stochastic update**: Add noise to represent biological uncertainty
  - `network.update_noise()`
- **Attractor analysis**: Find steady states and limit cycles
  - `SteadyStateCalculator`

### 2. Probabilistic Boolean network

Basic operations are similar to BNs.

- **Network loading**: Load networks from text files, or string representations
  - `load_pbn_from_file`, `load_pbn_from_string`
- **Stochastic Simulation**: Run Monte Carlo simulations over steps
  - `pbn.update_noise()`
- **Attractor analysis**: Find steady states (state distributions) via Monte Carlo or Two-state Markov Chain (TSMC) methods
  - `SteadyStateCalculator`

### 3. Knowledge Graph (KG) integration

BNMPy can build and extend models using the SIGNOR knowledge graph.

* **Build KG-derived BN**: Build a Boolean Network (BN) directly from a Knowledge Graph (KG)

  * `load_signor_network`
* **Merge models**: Merge an original BN with a KG-derived model into either a BN or a PBN

  * Deterministic: `merge_networks`
  * Probabilistic (PBN): `merge_networks`
* **Targeted extension to PBN**: Extend an existing BN by adding nodes and rules informed by KG

  *  `extend_networks`

### 4. PBN Optimization

#### Parameter optimization

- **Objective function**: MSE-based optimization for experimental data fitting
- **Optimization algorithms**: Differential evolution, particle swarm optimization
  - `ParameterOptimizer`
- **Discrete optimization**: Support for discrete parameter spaces
- **Result Evaluation**: Compare predictions with experimental observations
  - `SimulationEvaluator`, `ExperimentData`, `extract_experiment_nodes`, `generate_experiments`, `evaluate_optimization_result`, `evaluate_pbn`
- **Model compression**: Simplify models while preserving behavior
  - `compress_model`

## Examples

Tutorials:

- **[BN_simulation.ipynb](./Examples/BN_simulation.ipynb)**: Basic Boolean network simulation
- **[PBN_simulation.ipynb](./Examples/PBN_simulation.ipynb)**: Probabilistic Boolean network simulation
- **[knowledge_graph.ipynb](./Examples/knowledge_graph.ipynb)**: Build from SIGNOR, extend, and merge into BN/PBN
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
# initial state (array or dictionary)
x0 = [0, 0, 0, 0, 0, 0]  # array: order matches gene order in string
# or
x0 = {'A': 0, 'B': 1, 'C': 0, 'D': 0, 'E': 0, 'F': 0}  # dict: keys are gene names
network = BNMPy.load_network_from_string(network_string, initial_state=x0)

# Visualize the network
BNMPy.vis_network(network, output_html="SimpleBN.html", interactive=True)

# Simulate with noise = 0.05 for 10 steps
network.update_noise(p=0.05, iterations=10)

# Calculate steady states
calc = BNMPy.SteadyStateCalculator(network)
steady_state = calc.compute_steady_state(n_runs=20,n_steps=10000)

```

Note that when assigning initial states or obtaining steady states, gene order matches their order in the network string/file, and can be obtained by calling `network.nodeDict`. Currently it support both array input and dictionary input (genes not specified default to 0.)

## Documentation

- **Functions**: [https://ilyalab.github.io/BNMPy/](https://ilyalab.github.io/BNMPy/) (in development)
- **Tutorials**: Check the [Examples](./Examples/) directory for tutorials
- **Optimization Guide**: See [Optimizer README](./src/Optimizer_README.md) for advanced optimization features
