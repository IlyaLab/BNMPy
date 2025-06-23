import numpy as np
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass
from scipy.optimize import differential_evolution, minimize, OptimizeResult
import pyswarms as ps
import copy
import sys
from .simulation_evaluator import SimulationEvaluator
from .experiment_data import ExperimentData

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    success: bool
    message: str
    x: np.ndarray  # Best parameters
    fun: float  # Best objective value
    nfev: int  # Number of function evaluations
    nit: int  # Number of iterations
    method: str  # Optimization method used
    cij_matrix: np.ndarray  # Reshaped parameters as Cij matrix
    convergence_info: Dict[str, Any]  # Additional convergence information

# Custom exception for optimization errors
class OptimizationError(Exception):
    pass

class ParameterOptimizer:
    """
    PBN Parameter Optimizer
    """
    
    def __init__(self, pbn, experiments, config=None, nodes_to_optimize=None, discrete=False, verbose=False):
        """
        Initialize the optimizer
        
        Parameters:
        -----------
        pbn : ProbabilisticBN
            A BNMPy PBN object
        experiments : list or Path
            List of experiment dictionaries or path to csv file
        config : dict, optional
            Configuration for the optimizer
        nodes_to_optimize : list, optional
            List of node names to optimize. If None, all nodes are optimized.
        discrete : bool, optional
            If True, performs discrete optimization by selecting the best function.
        verbose : bool, optional
            If True, enables verbose output during optimization.
        """
        self.pbn = pbn
        self.config = self._default_config()
        if config:
            self.config.update(config)
        if isinstance(experiments, str):
            experiments = ExperimentData.load_from_csv(experiments)
        self.evaluator = SimulationEvaluator(pbn, experiments, self.config, nodes_to_optimize)
        self.discrete = discrete
        self.verbose = verbose
        self._optimization_history = []
        
        # For progress tracking
        self._iteration_count = 0
        self._last_best_score = float('inf')
        self._last_improvement = 0
        self._stagnation_counter = 0
        self._max_iter = 0  # For progress bar

    def _default_config(self) -> Dict:
        """Default optimization configuration"""
        return {
            'de_params': {
                'strategy': 'best1bin',
                'maxiter': 500,
                'popsize': 15,
                'tol': 0.01,
                'mutation': (0.5, 1),
                'recombination': 0.7,
                'seed': None,
                'disp': False,
                'init': 'sobol',
                'updating': 'deferred',
                'workers': -1
            },
            'max_try': 1,  # Maximum number of attempts if optimization fails
            'pso_params': {
                'n_particles': 30,
                'iters': 100,
                'options': {'c1': 0.5, 'c2': 0.3, 'w': 0.9},
                'ftol': 1e-3,  # Function tolerance for early stopping
                'ftol_iter': 10  # Number of iterations to check for stagnation
            },
            'local_params': {
                'method': 'L-BFGS-B',
                'options': {
                    'disp': False,
                    'maxiter': 500
                }
            },
            'discrete_params': {
                'threshold': 0.5
            },
            'success_threshold': 0.01,  # An SSE below this is always a success
            'early_stopping': False  # Control early stopping for both DE and PSO
        }
        
    def optimize(self, method: str = 'differential_evolution'):
        """
        Run the optimization
        
        Parameters:
        -----------
        method : str
            Optimization method to use ('differential_evolution', 'particle_swarm', 'local')
            
        Returns:
        --------
        OptimizeResult
            The optimization result object from scipy.optimize
        """
        bounds = self.evaluator.get_parameter_bounds()
        if not bounds:
            raise ValueError("No parameters to optimize. Check nodes_to_optimize.")

        # Get max_try configuration
        max_try = self.config.get('max_try', 1)
        success_threshold = self.config.get('success_threshold', 0.01)
        
        print(f"\nRunning optimization using method: {method}")
        print(f"Maximum attempts: {max_try}")

        for attempt in range(max_try):
            print(f"\n--- Attempt {attempt+1}/{max_try} ---")
            try:
                result = self._run_single_optimization(method, bounds)

                # A run is successful if the optimizer says so OR if the SSE is below our threshold
                is_successful = result.success or result.fun < success_threshold

                if is_successful:
                    best_score = result.fun
                    best_result = result
                    best_result.success = True 
                    print(f"\nSuccessful optimization found in attempt {attempt + 1}")
                    print(f"  - SSE: {best_score:.6f}")
                    if not result.success:
                        print(f"  (Run considered successful based on SSE < {success_threshold})")
                    break  # Stop trying if we have a successful result

                else:
                    print(f"Optimization attempt {attempt + 1} did not succeed: {result.message} (SSE: {result.fun:.6f})")
                    if result.fun < best_score:
                        best_score = result.fun
                        best_result = result

            except Exception as e:
                print(f"An error occurred during optimization attempt {attempt + 1}: {e}")

        if best_result is None:
            raise OptimizationError("All optimization attempts failed.")

        print(f"\n--- Optimization finished. Best SSE found: {best_result.fun:.6f} ---")

        # Update self.pbn with the final parameters 
        final_params = best_result.x
        final_cij = self.evaluator._vector_to_cij_matrix(final_params)

        # If discrete mode, discretize the final result and update the PBN again
        if self.discrete:
            print("\nDiscretizing final parameters...")
            final_cij = self._discretize_cij(final_cij)
            print("\nDiscretized Selection Probabilities:")
            print(final_cij)
        
        self.evaluator._update_pbn_parameters(final_cij)
        
        # Print the final PBN rules in the desired format
        pbn_rules = self.get_pbn_rules_string()
        print("\n--- Optimized PBN Rules ---")
        print(pbn_rules)
        print("---------------------------\n")

        return best_result

    def _run_single_optimization(self, method: str, bounds: List[tuple]):
        """Runs a single instance of the specified optimization method."""
        self._iteration_count = 0
        self._last_best_score = float('inf')
        self._stagnation_counter = 0
        self._optimization_history = [] # Reset history for each new run
        
        if method == 'differential_evolution':
            de_params = self.config['de_params'].copy()
            self._max_iter = de_params.get('maxiter', 500)
            de_params['callback'] = self._de_callback
            result = differential_evolution(self.evaluator.objective_function, bounds, **de_params)
            result.history = self._optimization_history
            result.method = method
            # Clear the progress line
            if not self.verbose:
                sys.stdout.write('\n')
                sys.stdout.flush()
            return result
        elif method == 'particle_swarm':
            result = self._pso_optimizer(bounds)
            result.method = method
            return result
        elif method == 'local':
            local_params = self.config['local_params'].copy()
            x0 = np.random.rand(len(bounds)) # Random initial guess
            result = minimize(self.evaluator.objective_function, x0, bounds=bounds, **local_params)
            result.history = self._optimization_history
            result.method = method
            return result
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _de_callback(self, xk, convergence, **kwargs):
        """Callback for DE to monitor progress with a single updating line and progress bar"""
        self._iteration_count += 1
        current_score = self.evaluator.objective_function(xk)
        self._optimization_history.append(current_score)
        
        if current_score < self._last_best_score:
            self._last_best_score = current_score

        # Calculate progress percentage
        progress = min(100.0, (self._iteration_count / self._max_iter) * 100)
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Update single line with progress
        if not self.verbose:
            sys.stdout.write(f'\r[{bar}] {progress:.1f}% | Iter: {self._iteration_count}/{self._max_iter} | Best SSE: {self._last_best_score:.6f} | Conv: {convergence:.4f}')
            sys.stdout.flush()
        else:
            # In verbose mode, still print each iteration
            print(f"Iteration {self._iteration_count}: Current SSE: {current_score:.6f}, Best SSE so far: {self._last_best_score:.6f}, Convergence: {convergence:.4f}")

        # Early stopping logic
        early_stopping = self.config.get('early_stopping', False)
        if early_stopping:
            success_threshold = self.config.get('success_threshold', 0.01)
            if current_score < success_threshold:
                if self.verbose: 
                    print(f"\nEarly stopping: SSE below success threshold: {success_threshold}")
                return True

    def _pso_optimizer(self, bounds):
        """PSO implementation using pyswarms library with early stopping support"""
        pso_params = self.config['pso_params'].copy()
        
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        pyswarms_bounds = (lb, ub)

        def pso_objective(x):
            return np.array([self.evaluator.objective_function(p) for p in x])

        # Handle early stopping by setting ftol
        early_stopping = self.config.get('early_stopping', False)
        if not early_stopping:
            pso_params['ftol'] = -np.inf  # Disable early stopping
            pso_params['ftol_iter'] = 1

        # Extract parameters for GlobalBestPSO constructor
        n_particles = pso_params.get('n_particles', 30)
        iters = pso_params.get('iters', 100)
        
        # optimize parameters
        optimize_kwargs = {'iters': iters}
            
        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=len(bounds),
            options=pso_params.get('options', {'c1': 0.5, 'c2': 0.3, 'w': 0.9}),
            bounds=pyswarms_bounds,
            ftol=pso_params.get('ftol', 1e-3),
            ftol_iter=pso_params.get('ftol_iter', 10)
        )
        
        cost, pos = optimizer.optimize(pso_objective, **optimize_kwargs)

        # Check if early stopping occurred
        success_threshold = self.config.get('success_threshold', 0.01)
        early_stopped = len(optimizer.cost_history) < iters
        
        result = OptimizeResult(
            fun=cost,
            x=pos,
            success=cost < success_threshold,
            message=f"pyswarms optimization finished{' (early stopped)' if early_stopped else ''}.",
            nfev=len(optimizer.cost_history) * n_particles,
            nit=len(optimizer.cost_history)
        )
        result.history = optimizer.cost_history
        return result

    def _discretize_cij(self, cij_matrix: np.ndarray) -> np.ndarray:
        """
        Discretize Cij matrix by selecting the function with the highest probability,
        respecting a threshold.
        """
        discrete_params = self.config.get('discrete_params', {})
        threshold = discrete_params.get('threshold', 0.5)

        discretized_matrix = cij_matrix.copy()
        for node_idx in self.evaluator.node_indices_to_optimize:
            num_functions = self.pbn.nf[node_idx]
            if num_functions > 0:
                row = discretized_matrix[node_idx, :num_functions]
                max_prob_idx = np.argmax(row)
                new_row = np.zeros(num_functions)
                # Only select the function if its probability is above the threshold
                if row[max_prob_idx] >= threshold:
                    new_row[max_prob_idx] = 1.0
                else:
                    # If no function meets the threshold, the result remains the
                    # highest probability one, but this indicates uncertainty.
                    # We still pick the best one to ensure a valid discrete state.
                    new_row[max_prob_idx] = 1.0
                    if self.verbose: print(f"  - Discretization warning: Highest probability for node {list(self.pbn.nodeDict.keys())[node_idx]} ({row[max_prob_idx]:.2f}) is below threshold ({threshold}).")

                padded_new_row = np.full(cij_matrix.shape[1], -1.0)
                padded_new_row[:num_functions] = new_row
                discretized_matrix[node_idx, :] = padded_new_row
        return discretized_matrix

    def _detect_stagnation_periods(self, history: np.ndarray, tolerance: float = 1e-6) -> List[tuple]:
        """Detect periods where optimization stagnated"""
        if len(history) < 10:
            return []
        
        stagnation_periods = []
        current_start = None
        
        for i in range(1, len(history)):
            if abs(history[i] - history[i-1]) < tolerance:
                if current_start is None:
                    current_start = i-1
            else:
                if current_start is not None and i - current_start > 5:  # At least 5 iterations of stagnation
                    stagnation_periods.append((current_start, i-1))
                current_start = None
        
        # Check if we ended in a stagnation period
        if current_start is not None and len(history) - current_start > 5:
            stagnation_periods.append((current_start, len(history)-1))
        
        return stagnation_periods

    def plot_optimization_history(self, result: OptimizeResult, save_path: Optional[str] = None, 
                                 show_stagnation: bool = False, log_scale: bool = False):
        """
        Plot the optimization history (SSE over iterations).
        
        Parameters:
        -----------
        result : OptimizeResult
            The optimization result object
        save_path : str, optional
            Path to save the plot. If None, displays the plot.
        show_stagnation : bool, optional
            Whether to highlight stagnation periods
        log_scale : bool, optional
            Whether to use logarithmic scale for y-axis
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Install with: pip install matplotlib")
            return
        
        if not hasattr(result, 'history') or not result.history:
            print("No optimization history available for plotting.")
            return
        
        history = np.array(result.history)
        
        plt.figure(figsize=(10, 6))
        plt.plot(history, 'b-', linewidth=2, label='SSE')
        
        if log_scale:
            plt.yscale('log')
        
        if show_stagnation:
            stagnation_periods = self._detect_stagnation_periods(history)
            for start, end in stagnation_periods:
                plt.axvspan(start, end, alpha=0.3, color='red', label='Stagnation' if start == stagnation_periods[0][0] else "")
        
        plt.xlabel('Iteration')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.title(f'Optimization History (Method: {result.get("method", "unknown")})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add text box with final statistics
        stats_text = f'Final SSE: {result.fun:.6f}\nIterations: {result.nit}\nSuccess: {result.success}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def get_pbn_rules_string(self) -> str:
        """
        Format the final PBN into a readable string.
        """
        if not hasattr(self.pbn, 'gene_functions'):
            return "Could not format PBN rules: function strings not found in PBN object."

        rules_string = []
        node_names = sorted(self.pbn.nodeDict.keys(), key=lambda k: self.pbn.nodeDict[k])
        
        for node_name in node_names:
            node_idx = self.pbn.nodeDict[node_name]
            if node_name in self.pbn.gene_functions:
                functions = self.pbn.gene_functions[node_name]
                probabilities = self.pbn.cij[node_idx, :]
                
                for i, func in enumerate(functions):
                    prob = probabilities[i]
                    if prob > 1e-6 or (len(functions) == 1 and np.isclose(prob, 1.0)):
                        rules_string.append(f"{node_name} = {func}, {prob:.4f}")
                        
        return "\n".join(rules_string)

    def get_optimized_pbn(self, result: OptimizeResult) -> Any:
        """
        Get a new PBN object with the optimized parameters from a result.
        
        Parameters:
        -----------
        result : OptimizationResult
            The result object from a successful optimization run.
            
        Returns:
        --------
        ProbabilisticBN
            A new PBN object with the optimized parameters.
        """
        # Create a deep copy to avoid modifying the original PBN object
        optimized_pbn = copy.deepcopy(self.pbn)
        
        # Get the Cij matrix from the result parameters
        cij_matrix = self.evaluator._vector_to_cij_matrix(result.x)
        
        # Discretize if the optimization was run in discrete mode
        if self.discrete:
            cij_matrix = self._discretize_cij(cij_matrix)
        
        # Update the new PBN with the final Cij matrix
        optimized_pbn.cij = cij_matrix
        if hasattr(optimized_pbn, 'update_cumulative_probabilities'):
            optimized_pbn.update_cumulative_probabilities()
            
        return optimized_pbn 