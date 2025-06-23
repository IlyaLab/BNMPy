# BNMPy Optimizer

A Python implementation of PBN parameter optimization based on the optPBN framework. This module enables parameter estimation for Probabilistic Boolean Networks (PBNs) using experimental data.

## Installation

The optimizer is part of BNMPy. Required dependencies:

```bash
pip install numpy scipy pandas pyswarms matplotlib
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
Experiments,Stimuli,Inhibitors,Measured_nodes,Measured_values
1,TGFa,TNFa,"NFkB,ERK,C8,Akt","0.7,0.88,0,1"
2,TNFa,TGFa,"NFkB,ERK,C8,Akt","0.3,0.12,1,0"
```

- `Stimuli`: Nodes fixed to 1 (comma-separated)
- `Inhibitors`: Nodes fixed to 0 (comma-separated)
- `Measured_nodes`: Nodes with experimental measurements
- `Measured_values`: Corresponding values (0-1, normalized)

#### PBN Data

A ProbabilisticBN, see BNMPy.

### Configurations

Current support two optimization methods:

* Differential evolution
* Particle Swarm Optimization

The parameters of each approach can be set via a config dictionary:

```python
config = {
# Early stopping control
    'early_stopping': True,           # Enable early stopping for both DE and PSO
    'success_threshold': 0.005,       # Stop if SSE drops below this value    # Retry control
    'max_try': 3,                     # Try up to 3 times if optimization fails  

# Differential Evolution parameters
    'de_params': {
        'strategy': 'best1bin',
        'maxiter': 500,
        'popsize': 15,
        'tol': 0.01,
        'mutation': (0.5, 1),
        'recombination': 0.7,
        'init': 'sobol',
        'updating': 'deferred',
        'workers': -1, # Use all available cores for parallelization
        'polish': False  # Disable polish step for faster runs
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

Control when optimization should stop early:

- **For DE**: Uses `success_threshold` to stop when SSE drops below threshold
- **For PSO**: Uses both `success_threshold` and function tolerance (`ftol`, `ftol_iter`)
- Enable with `early_stopping: True` in config

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
- `fun`: Final objective value (SSE)
- `history`: List of best SSE values at each iteration
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

## References

Based on the optPBN framework:

- Trairatphisan, P., et al. (2014). "optPBN: An Optimisation Toolbox for Probabilistic Boolean Networks." PLOS ONE 9(7): e98001.
