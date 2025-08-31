# BNMPy: Boolean network modeling and optimization

A Python library for Boolean Network (BN) and Probabilistic Boolean Network (PBN) modeling, simulation, optimization, and analysis with applications in systems biology.

## Installation

To install, run `pip install -e .` in this directory.

## Core functionalities

### 1. Boolean network simulation (`BNMPy/`)

#### Basic operations

- **Network loading**: Load networks from text files, SBML files, or string representations
- **Network construction**: Build networks from connectivity matrices and Boolean functions
- **Network manipulation**: Knockout/knockdown specific nodes, rename nodes, merge networks
- **Network visualization**: Interactive network visualizations with igraph and matplotlib

#### Simulation

- **Deterministic update**: Synchronous and asynchronous update schemes
- **Stochastic update**: Add noise to represent biological uncertainty
- **Attractor analysis**: Find steady states and limit cycles

### 2. Probabilistic Boolean network (`BNMPy/`)

#### PBN Features

- **Network loading**: Load networks from text files, or string representations
- **Stochastic Simulation**: Run Monte Carlo simulations over steps
- **Attractor analysis**: Find steady states (state distributions) via Monte Carlo or Two-state Markov Chain (TSMC) methods

### 3. PBN Optimization (`Optimizer/`)

#### Parameter optimization

- **Objective function**: MSE-based optimization for experimental data fitting
- **Optimization algorithms**: Differential evolution, particle swarm optimization
- **Discrete optimization**: Support for discrete parameter spaces
- **Result Evaluation**: Compare predictions with experimental observations

#### Model Compression

- **Node removal**: Remove non-observable and non-controllable nodes
- **Path collapse**: Collapse linear paths to simplify network structure

#### Sensitivity Analysis

- **Morris method**: One-at-a-time sensitivity analysis
- **Sobol methods**: Global sensitivity analysis
- **Influence analysis**: Node influence on network behavior
- **Parameter sensitivity**: Identify most critical model parameters

## Examples

- **[BN_simulation.ipynb](./Examples/BN_simulation.ipynb)**: Basic Boolean network simulation
- **[PBN_simulation.ipynb](./Examples/PBN_simulation.ipynb)**: Probabilistic Boolean network simulation
- **[BN_PBN_steady_state.ipynb](./Examples/BN_PBN_steady_state.ipynb)**: Steady state analysis
- **[BN_compression.ipynb](./Examples/BN_compression.ipynb)**: Model compression and simplification
- **[Optimization.ipynb](./Examples/Optimization.ipynb)**: Parameter optimization with experimental data
- **[workflow_example.ipynb](./Examples/workflow_example.ipynb)**: Complete workflow from data to optimized model

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
# initial state
x0 = [0, 0, 0, 0, 0, 0]
network = BNMPy.load_network_from_string(network_string, initial_state=x0)

# Visualize the network
BNMPy.vis_network(network, output_html="SimpleBN.html", interactive=True)

# Simulate with noise = 0.05 for 10 steps
network.update_noise(p=0.05, iterations=10)

# Calculate steady states
calc = BNMPy.SteadyStateCalculator(network)
steady_state = calc.compute_steady_state(n_runs=20,n_steps=10000)

```

## Documentation

- **Tutorials**: Check the [Examples](./Examples/) directory for tutorials
- **Optimization Guide**: See [Optimizer README](./src/Optimizer/README.md) for advanced optimization features
