# SteadyStateCalculator for BNMPy

## Overview

The `SteadyStateCalculator` class provides steady-state calculation capabilities for both Boolean Networks and Probabilistic Boolean Networks in BNMPy.

## Quick Start

```python
import BNMPy
import numpy as np

# Load a PBN
network_string = """
N1 = N1, 1
N2 = N2, 1
N3 = N1, 0.6
N3 = N1 & !N2, 0.4
"""
x0 = np.array([1, 1, 0])
network = BNMPy.load_network(network_string, initial_state=x0)

# Create calculator
calc = BNMPy.SteadyStateCalculator(network)

# Calculate steady state
steady_state = calc.compute_steady_state(method='monte_carlo', n_runs=20, n_steps=10000, p_noise=0.05)
print(f"Steady state: {steady_state}")
```

## Methods

The `compute_steady_state()` method automatically selects the appropriate calculation method:

- For Boolean Networks: uses deterministic attractor finding
- For PBNs: uses Monte Carlo or TSMC based on the `method` parameter

**Parameters**:

- `method` (str): 'monte_carlo' (default), 'tsmc', or 'deterministic'
- `**kwargs`: Method-specific parameters

**Returns**: numpy array of steady-state probabilities (one value per node)

---

### 1. TSMC Method (Two-State Markov Chain)

**Purpose**: Calculate steady state using transition probability analysis.

**When to use**:

- When you need faster calculations
- For networks with well-defined transition properties

**Parameters**:

- `epsilon` (float, default=0.001): Range of transition probability (smaller = more accurate)
- `r` (float, default=0.025): Range of accuracy (smaller = more accurate)
- `s` (float, default=0.95): Probability of accuracy (closer to 1 = more confident)
- `p_noise` (float, default=0): Noise probability
- `p_mir` (float, default=0.001): Perturbation probability (Miranda-Parga scheme)
- `initial_nsteps` (int, default=100): Initial number of simulation steps
- `max_iterations` (int, default=500): Maximum convergence iterations
- `freeze_constant` (bool, default=False): Whether to freeze constant nodes
- `seed` (int, optional): Random seed for reproducibility

**How it works**:

1. Runs simulation and analyzes state transitions
2. Automatically determines required burn-in period
3. Adjusts simulation length based on convergence criteria
4. Ensures results meet specified statistical accuracy

**The `freeze_constant` parameter**:

In the original MATLAB implementation, self-loop nodes (e.g., `A = A`) are treated as inputs and never perturbed. However, this causes different behavior between TSMC and Monte Carlo:

- **Monte Carlo**: Each run starts with a fresh random state, so self-loops naturally converge to 0.5
- **TSMC**: Uses one initial state for the entire trajectory, so self-loops stay at their initial value

Setting `freeze_constant=False` allows TSMC to perturb self-loop nodes, making the two methods converge to identical stationary distributions.

**Example**:

```python
# Similar to Monte Carlo behavior (recommended)
steady_state = calc.compute_steady_state(
    method='tsmc',
    r=0.01,
    p_mir=0.01,  # Adds perturbation to match noise behavior
    initial_nsteps=100,
    max_iterations=5000,
    freeze_constant=False,  # Allow self-loops to be perturbed
    seed=9
)

# Original MATLAB behavior
steady_state = calc.compute_steady_state(
    method='tsmc',
    r=0.01,
    initial_nsteps=100,
    max_iterations=5000,
    freeze_constant=True,  # Keep self-loops fixed
    seed=9
)
```

---

### 2. Monte Carlo Method

**Purpose**: Calculate steady state using multiple independent simulations.

**When to use**:

- Default method for most PBN steady-state calculations
- When you need reliable estimates with convergence checking

**Parameters**:

- `n_runs` (int, default=10): Number of independent simulation runs
- `n_steps` (int, default=1000): Number of simulation steps per run
- `p_noise` (float, default=0): Noise probability for state flipping
- `analyze_convergence` (bool, default=False): Whether to analyze and plot convergence
- `output_node` (str, optional): Specific node for convergence analysis
- `seed` (int, optional): Random seed for reproducibility

**How it works**:

1. Runs multiple independent simulations from random initial states
2. For each run, takes the second half of the trajectory as steady state
3. Averages across all runs to get final steady-state probabilities

**Example**:

```python
# Basic usage
steady_state = calc.compute_steady_state(
    method='monte_carlo',
    n_runs=20,
    n_steps=10000,
    p_noise=0.05,
    seed=9
)

# With convergence analysis
steady_state, convergence_info = calc.compute_steady_state(
    method='monte_carlo',
    n_runs=20,
    n_steps=20000,
    p_noise=0.05,
    analyze_convergence=True,
    output_node='N3',
    seed=9
)
```

**Convergence Analysis**: When `analyze_convergence=True`, the method will:

- Calculate running averages at different time points
- Compute relative changes between consecutive averages
- Display a convergence plot
- Return both the steady state and convergence information

---

### 3. Deterministic Method

**Purpose**: Find attractors (fixed points and limit cycles) in Boolean networks.

**Parameters**:

- `n_runs` (int, default=100): Number of random initial conditions to try
- `n_steps` (int, default=1000): Maximum steps before declaring no cycle found
- `verbose` (bool, default=True): Whether to print attractor information
- `seed` (int, optional): Random seed for reproducibility

**How it works**:

1. Tries multiple random initial conditions
2. Simulates until finding a repeating pattern
3. Identifies fixed points (period 1) and limit cycles (period > 1)
4. Returns all unique attractors found

**Returns**: Dictionary with:

- `fixed_points`: List of fixed point states
- `cyclic_attractors`: List of limit cycles (each cycle is a list of states)

**Example**:

```python
import BNMPy

# Load a Boolean network
network = BNMPy.load_network("network.txt")
calc = BNMPy.SteadyStateCalculator(network)

# Find attractors
attractors = calc.compute_steady_state(n_runs=100, n_steps=1000, verbose=True)

# Access results
fixed_points = attractors['fixed_points']
cycles = attractors['cyclic_attractors']

print(f"Found {len(fixed_points)} fixed points")
print(f"Found {len(cycles)} limit cycles")
```

---

## Experimental Conditions

### Setting Perturbations

You can simulate experimental conditions by setting stimuli (activators) and inhibitors:

**Parameters**:

- `stimuli` (list): Node names to fix at value 1
- `stimuli_efficacy` (list, optional): Efficacy values 0-1 (default: 1.0 for all)
- `inhibitors` (list): Node names to fix at value 0
- `inhibitors_efficacy` (list, optional): Efficacy values 0-1 (default: 1.0 for all)

**Efficacy values**:

- `1.0` (default): Full knockout/stimulation
- `< 1.0`: Partial efficacy - creates probabilistic perturbation
  - For inhibitors: P(node=0) = efficacy
  - For stimuli: P(node=1) = efficacy

**Note**: Efficacy option is only available for Monte Carlo method.

### Example

```python
import BNMPy
import numpy as np

# Load network
network_string = """
N1 = N1, 1
N2 = N2, 1
N3 = N1, 0.6
N3 = N1 & !N2, 0.4
"""
network = BNMPy.load_pbn_from_string(network_string, initial_state=[1, 1, 0])
calc = BNMPy.SteadyStateCalculator(network)

# 1. Stimulate N1 (fix to 1)
calc.set_experimental_conditions(stimuli=['N1'])
ss1 = calc.compute_steady_state(method='monte_carlo', n_runs=20, n_steps=10000, p_noise=0)
print(f"N1 stimulated: {ss1}")  # [1., 0., 1.]

# 2. Stimulate N2 (fix to 1)
calc.set_experimental_conditions(stimuli=['N2'])
ss2 = calc.compute_steady_state(method='monte_carlo', n_runs=20, n_steps=10000, p_noise=0)
print(f"N2 stimulated: {ss2}")  # [0., 1., 0.]

# 3. Partial efficacy perturbations
calc.set_experimental_conditions(
    stimuli=['N1'],
    stimuli_efficacy=[0.5],
    inhibitors=['N2'],
    inhibitors_efficacy=[0.5]
)
ss3 = calc.compute_steady_state(method='monte_carlo', n_runs=20, n_steps=10000, p_noise=0.05, seed=9)
print(f"Partial efficacy: {ss3}")

# Always reset when done
calc.reset_network_conditions()
```

## See Also

- **Examples**: See `Examples/BN_PBN_steady_state.ipynb` for detailed examples
- **Optimization**: See `src/Optimizer_README.md` for integration with parameter optimization

## References

- Trairatphisan, P., et al. (2014). "optPBN: An Optimisation Toolbox for Probabilistic Boolean Networks." PLOS ONE 9(7): e98001.
- Shmulevich, I., & Dougherty, E. R. (2010). Probabilistic Boolean Networks: The Modeling and Control of Gene Regulatory Networks.
- Miranda, E., & Parga, N. (2010). "Noise and low-level dynamics can coordinate multisite activity in the mammalian olfactory bulb." Journal of Neurophysiology.
