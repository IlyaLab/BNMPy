# BNMPy Optimizer

A Python implementation of PBN parameter optimization based on the optPBN framework. This module enables parameter estimation for Probabilistic Boolean Networks (PBNs) using experimental data.

## Installation

The optimizer is part of BNMPy. Required dependencies:

```bash
pip install numpy scipy pandas pyswarms matplotlib networkx
```

## Usage

### Basic Usage

```python
from BNMPy.Optimizer import ParameterOptimizer
from BNMPy.BMatrix import load_pbn_from_string

# Load or create your PBN
pbn = load_pbn_from_string(...)

# Initialize optimizer
optimizer = ParameterOptimizer(pbn, "experiments.csv", nodes_to_optimize=['Cas3'], verbose=False)

# Run optimization
result = optimizer.optimize(method='differential_evolution')

# Get an optimized PBN object
if result.success:
    optimized_pbn = optimizer.get_optimized_pbn(result)
```

### Input Data Format

#### Experimental Data

CSV file format:

```csv
Experiments,Stimuli,Stimuli_efficacy,Inhibitors,Inhibitors_efficacy,Measured_nodes,Measured_values
1,TGFa,1,TNFa,1,"NFkB,ERK,C8,Akt","0.7,0.88,0,1"
2,TNFa,1,TGFa,1,"NFkB,ERK,C8,Akt","0.3,0.12,1,0"
3,"TGFa,TNFa","1,1",,,"NFkB,ERK,C8,Akt","1,1,1,1"
4,"TGFa,TNFa","1,1",PI3K,0.7,"NFkB,ERK,C8,Akt","0.3,0.12,1,0"
```

- `Stimuli`: Nodes fixed to 1 (comma-separated)
- `Stimuli_efficacy`: Efficacy of stimuli (0-1, comma-separated, optional)
- `Inhibitors`: Nodes fixed to 0 (comma-separated)
- `Inhibitors_efficacy`: Efficacy of inhibitors (0-1, comma-separated, optional)
- `Measured_nodes`: Nodes with experimental measurements
- `Measured_values`: Corresponding values (0-1, normalized)

**Efficacy Values:**

- **1.0 (default)**: Full efficacy - node is completely knocked out/stimulated
- **< 1.0**: Partial efficacy - creates probabilistic perturbation
  - **For inhibitors (target=0)**: P(node=0) = efficacy, P(node=1) = 1-efficacy
  - **For stimuli (target=1)**: P(node=1) = efficacy, P(node=0) = 1-efficacy
- **Example**: `PI3K,0.7` means PI3K inhibition has 70% probability of setting PI3K=0, 30% of PI3K=1
- If efficacy columns are empty, defaults to 1.0 for all perturbations

#### PBN Data

A ProbabilisticBN, see BNMPy.

### Configurations

Current support two optimization methods:

* Differential evolution
* Particle Swarm Optimization

The parameters of each approach can be set via a config dictionary:

```python
config = {
# Global settings
    'seed': 9,                        # Global seed for random number generation (affects all stochastic processes)
    'success_threshold': 0.005,       # Global success threshold for final result evaluation
    'max_try': 3,                     # Try up to 3 times if optimization fails
    'display_rules_every': 50,        # Display optimized rules every N iterations (0 = disabled)

# Differential Evolution parameters
    'de_params': {
        'strategy': 'best1bin',
        'maxiter': 500,
        'popsize': 15,
        'tol': 0.01,                  # Relative tolerance for scipy DE convergence
        'atol': 0,                    # Absolute tolerance for scipy DE convergence
        'mutation': (0.5, 1),
        'recombination': 0.7,
        'init': 'sobol',
        'updating': 'deferred',
        'workers': -1,                # Use all available cores for parallelization
        'polish': False,              # Disable polish step for faster runs
        'early_stopping': True,       # Enable early stopping for DE
        'success_threshold': 0.01     # MSE threshold for DE early stopping
    },  

# Particle Swarm Optimization parameters
    'pso_params': {
        'n_particles': 30,
        'iters': 100,
        'options': {'c1': 0.5, 'c2': 0.3, 'w': 0.9},
        'ftol': 1e-6,                 # Function tolerance for early stopping
        'ftol_iter': 15,              # Check stagnation over this many iterations
    },  

# Steady state calculation
    'steady_state': {
        'method': 'monte_carlo',
        'monte_carlo_params': {
            'n_runs': 10,
            'n_steps': 1000
        }
    }
}

optimizer = ParameterOptimizer(pbn, "experiments.csv", config=config, verbose=False)
result = optimizer.optimize('differential_evolution')
```

#### Early Stopping

Early stopping behavior differs between optimization methods:

**Differential Evolution (DE):**

Two approaches:

- `early_stopping`: stop when MSE drops below `success_threshold`
- `tol` and `atol`: stops when `np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))`

**Particle Swarm Optimization (PSO):**

- `ftol`: Function tolerance - stops when fitness improvement is below this value
- `ftol_iter`: Number of iterations to check for stagnation

**Final Success Determination:**

- Both methods use the global `success_threshold` to determine if the final result is considered successful
- This evaluation happens after optimization completes, regardless of early stopping

#### Discrete Mode

For Boolean Network optimization:

```python
# Configure discrete optimization
config = {
    'discrete_params': {
        'threshold': 0.6
    }
}

# Run optimization in discrete mode
result = optimizer.optimize(
    method='differential_evolution',
    discrete=True
)
```

### Results

The optimization returns an enhanced `OptimizeResult` object containing:

- `success`: Boolean indicating if the optimizer terminated successfully
- `message`: Status message
- `x`: Optimized parameters (flattened vector)
- `fun`: Final objective value (MSE)
- `history`: List of best MSE values at each iteration
- `nfev`: Number of function evaluations
- `nit`: Number of iterations

### Plot Optimization History

```python
# Basic history plot
optimizer.plot_optimization_history(result)

# Advanced plotting with options
optimizer.plot_optimization_history(
    result, 
    save_path='optimization_history.png',
    show_stagnation=True,    # Highlight stagnation periods
    log_scale=True          # Use logarithmic scale
)
```

### Result Evaluation

Evaluate optimization results:

```python
from BNMPy.Optimizer import evaluate_optimization_result

evaluator = evaluate_optimization_result(
    result, 
    optimizer, 
    output_dir="evaluation_results",
    plot_residuals=True,
    save=True,
    detailed=True,
    figsize=(8, 6)
)
```

#### Evaluation Plots

The `evaluate_optimization_result` function generates several plots to assess optimization quality:

1. **Prediction vs Experimental Plot (`prediction_vs_experimental.png`)**
2. **Residuals Plot (`residuals.png`)**
3. **Optimization History Plot (`optimization_history.png`)**

## Model Compression

The Optimizer module includes model compression functionality to simplify Boolean Networks before optimization. Currently supports Boolean Networks only. This will allow to:

1. **Remove non-observable nodes**: Nodes that don't influence any measured species
2. **Remove non-controllable nodes**: Nodes that aren't influenced by any perturbed species
3. **Collapse linear paths**: Simplify cascades of intermediate nodes that don't branch
4. **Visualize compression results**: Show which nodes and edges were removed

### Example

Compression can automatically extract measured and perturbed nodes from experimental data:

```python
from BNMPy.Optimizer import ParameterOptimizer, extract_experiment_nodes
from BNMPy.Optimizer.model_compressor import compress_model
from BNMPy.BMatrix import load_network_from_file

# Load network and experimental data
network = load_network_from_file("network.txt")

# Extract nodes from experimental data
measured_nodes, perturbed_nodes = extract_experiment_nodes("experiments.csv")

# Compress network using experimental information
compressed_network, compression_info = compress_model(
    network,
    measured_nodes=measured_nodes,
    perturbed_nodes=perturbed_nodes
)

# Visualize compression results
from BNMPy.vis import vis_compression
vis_compression(
    network,
    compressed_network,
    compression_info,
    "compression_results.html"
)

# Run optimization on compressed network
optimizer = ParameterOptimizer(compressed_network, "experiments.csv")
result = optimizer.optimize(method='differential_evolution')
```

## Sensitivity Analysis

The Optimizer module includes sensitivity analysis tools to identify the most influential nodes affecting measurements:

```python
from BNMPy.Optimizer import sensitivity_analysis

# Run sensitivity analysis
results = sensitivity_analysis(
    network, 
    experiments, 
    config=config,  # similar config for the optimizer
    top_n=5
)

# Get top sensitive nodes
top_nodes = results['top_nodes']
sensitivity_df = results['sensitivity_df']
```

- **Morris Analysis**: One-at-a-time sensitivity analysis using Morris trajectories
- **Sobol Analysis**: Variance-based sensitivity analysis for comprehensive parameter importance
- **Simple Analysis**: Basic one-at-a-time approach for quick assessment

## References

Based on the optPBN framework:

- Trairatphisan, P., et al. (2014). "optPBN: An Optimisation Toolbox for Probabilistic Boolean Networks." PLOS ONE 9(7): e98001.
