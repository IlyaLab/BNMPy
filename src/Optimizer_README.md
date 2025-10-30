# BNMPy Optimizer

A Python implementation of Probabilistic Boolean Network (PBN) parameter optimization based on the optPBN framework. This module enables parameter estimation for PBNs using experimental data.

## Optimization

### Basic Usage

```python
import BNMPy

# Load or create your PBN
pbn = BNMPy.load_network(...)

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
- `Measured_nodes`: Nodes with experimental measurements OR a formula expression
- `Measured_values`: Corresponding values (0-1, normalized). For formulas, a single value per row is required

See [Examples/files/experiments_example.csv](../Examples/files/experiments_example.csv) for an example file.

**Efficacy Values:**

- **1.0 (default)**: Full efficacy - node is completely knocked out/stimulated
- **< 1.0**: Partial efficacy - creates probabilistic perturbation
  - **For inhibitors (target=0)**: P(node=0) = efficacy, P(node=1) = 1-efficacy
  - **For stimuli (target=1)**: P(node=1) = efficacy, P(node=0) = 1-efficacy
- **Example**: `PI3K,0.7` means PI3K inhibition has 70% probability of setting PI3K=0, 30% of PI3K=1
- If efficacy columns are empty, defaults to 1.0 for all perturbations

#### PBN Data

A ProbabilisticBN object, see [PBN_simulation.ipynb](../Examples/PBN_simulation.ipynb).

#### Formula-based Measurements (Phenotype score)

We can target an aggregate score instead of individual nodes using a formula based on node names and arithmetic operators. Two ways to use formulas:

- Place the formula directly in `Measured_nodes` and provide a single target value in `Measured_values`:

```csv
Experiments,Stimuli,Stimuli_efficacy,Inhibitors,Inhibitors_efficacy,Measured_nodes,Measured_values
1,,,,,"nodeA + nodeB - nodeC",0.75
```

- Or supply a global formula via the optimizer argument `Measured_formula` (overrides CSV `Measured_nodes` ), with each CSV row having a single `Measured_values` entry:

```python
optimizer = ParameterOptimizer(
    pbn,
    "experiments.csv",
    nodes_to_optimize=['nodeX'],
    verbose=False,
    Measured_formula="nodeA + nodeB - nodeC"
)
```

- Supported operators: `+`, `-`, `*`, `/`, parentheses and unary `+/-`
- Variables must correspond to nodes in the network (`pbn.nodeDict`)
- Measured values should be in the same range as the theoretical formula range
  - Formula range examples: `N1+N2+N3` → [0,3], `N1+N2-N3` → [-1,2], `N1-N2` → [-1,1]

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
- If unsuccessful, the optimizer will run up to `max_try` times.

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

The optimization returns an `OptimizeResult` object containing:

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
from BNMPy import evaluate_optimization_result

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

## References

Based on the optPBN framework:

- Trairatphisan, P., et al. (2014). "optPBN: An Optimisation Toolbox for Probabilistic Boolean Networks." PLOS ONE 9(7): e98001.
