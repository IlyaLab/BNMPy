import numpy as np
import pandas as pd
import time
import sys
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy.optimize import OptimizeResult
import pyswarms as ps
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

class OptimizationError(Exception):
    """Custom exception for optimization errors"""
    pass

class ParameterOptimizer:
    """
    Parameter optimization for Probabilistic Boolean Networks using experimental data.
    """
    
    def __init__(self, pbn, experiments, config=None, nodes_to_optimize=None, discrete=False, verbose=False):
        """
        Initialize the parameter optimizer.
        
        Parameters:
        -----------
        pbn : ProbabilisticBN
            BNMPy PBN object
        experiments : str or list
            Path to CSV file or list of experiment dictionaries
        config : dict, optional
            Configuration parameters for optimization
        nodes_to_optimize : list, optional
            List of node names to optimize. If None, optimize all nodes.
        discrete : bool, default=False
            Whether to perform discrete optimization
        verbose : bool, default=False
            Whether to print detailed optimization progress
        """
        self.pbn = pbn
        self.verbose = verbose
        self.discrete = discrete
        
        # Load experiments if string (CSV file path)
        if isinstance(experiments, str):
            self.experiments = ExperimentData.load_from_csv(experiments)
        else:
            self.experiments = experiments
        
        # Validate experiments
        ExperimentData.validate_experiments(self.experiments, pbn.nodeDict)
        
        # Initialize evaluator
        self.evaluator = SimulationEvaluator(pbn, self.experiments, config, nodes_to_optimize)
        
        # Set configuration
        self.config = config or self._default_config()
        
        # Initialize tracking variables
        self._iteration_count = 0
        self._max_iter = 0
        self._start_time = 0
        self._last_best_score = float('inf')
        self._optimization_history = []

    def _default_config(self) -> Dict:
        """Default configuration for optimization"""
        return {
            # Global settings
            'success_threshold': 0.005,       # Global success threshold for final result evaluation
            'max_try': 3,                     # Try up to 3 times if optimization fails
            'display_rules_every': 10,        # Display optimized rules every N iterations (0 = disabled)

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
                    'n_runs': 3,
                    'n_steps': 5000
                }
            }
        }

    def optimize(self, method: str = 'differential_evolution'):
        """
        Run parameter optimization using specified method.
        
        Parameters:
        -----------
        method : str, default='differential_evolution'
            Optimization method ('differential_evolution' or 'particle_swarm')
            
        Returns:
        --------
        OptimizeResult
            Optimization result object
        """
        if method not in ['differential_evolution', 'particle_swarm']:
            raise ValueError(f"Unsupported method: {method}. Use 'differential_evolution' or 'particle_swarm'")
        
        # Get parameter bounds
        bounds = self.evaluator.get_parameter_bounds()
        
        if not bounds:
            raise OptimizationError("No parameters to optimize")
        
        # Try optimization up to max_try times
        max_try = self.config.get('max_try', 3)
        best_result = None
        best_score = float('inf')
        
        for attempt in range(max_try):
            print(f"\nOptimization attempt {attempt + 1}/{max_try}")
            
            try:
                result = self._run_single_optimization(method, bounds)
                
                if result.fun < best_score:
                    best_result = result
                    best_score = result.fun
                
                # Check if we've achieved the success threshold
                success_threshold = self.config.get('success_threshold', 0.01)
                if result.fun < success_threshold:
                    if self.verbose:
                        print(f"Success threshold achieved: {result.fun:.6f} < {success_threshold}")
                    break
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_try - 1:
                    raise OptimizationError(f"All optimization attempts failed. Last error: {str(e)}")
        
        if best_result is None:
            raise OptimizationError("No successful optimization runs")

        print(f"\n--- Optimization finished. Best MSE found: {best_result.fun:.6f} ---")
        # Update self.pbn with the final parameters
        if hasattr(best_result, 'x') and best_result.x is not None:
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
        """Run a single optimization attempt"""
        self._start_time = time.time()
        self._last_best_score = float('inf')
        self._optimization_history = []
        
        if method == 'differential_evolution':
            return self._de_optimizer(bounds)
        elif method == 'particle_swarm':
            return self._pso_optimizer(bounds)
        else:
            raise ValueError(f"Unsupported method: {method}")

    def _de_optimizer(self, bounds):
        """Differential evolution implementation using scipy"""
        from scipy.optimize import differential_evolution
        print(f"Running DE optimization...")
        print("\n")
        if self.verbose:
            print(f"DE Setup:")
            print(f"  - Max iterations: {self.config['de_params']['maxiter']}")
            print(f"  - Population size: {self.config['de_params']['popsize']}")
            print(f"  - Tolerance: {self.config['de_params']['tol']}")
            print(f"  - Absolute tolerance: {self.config['de_params']['atol']}")
            print(f"  - Mutation: {self.config['de_params']['mutation']}")
            print(f"  - Recombination: {self.config['de_params']['recombination']}")
            print(f"  - Early stopping: {self.config['de_params']['early_stopping']}")
        
        de_params = self.config['de_params'].copy()
        de_params.pop('early_stopping', None)
        self._max_iter = de_params.get('maxiter', 500)
        
        # Run optimization
        result = differential_evolution(
            self.evaluator.objective_function,
            bounds,
            callback=self._de_callback,
            **de_params
        )
        
        # Clear any remaining display artifacts
        if not self.verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()
        
        # Check success based on threshold
        success_threshold = self.config.get('success_threshold', 0.01)
        success = result.fun < success_threshold
        
        return OptimizeResult(
            fun=result.fun,
            x=result.x,
            success=success,
            message=f"DE optimization finished with MSE: {result.fun:.6f}",
            nfev=result.nfev,
            nit=result.nit,
            history=self._optimization_history
        )

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
        
        # Check if we should display rules
        display_rules_every = self.config.get('display_rules_every', 0)
        should_display_rules = (display_rules_every > 0 and 
                               self._iteration_count % display_rules_every == 0)
        
        # display logic
        if should_display_rules:
            # Clear any progress bar if not in verbose mode
            if not self.verbose:
                sys.stdout.write('\r' + ' ' * 120 + '\r')
                sys.stdout.flush()
            
            elapsed_time = time.time() - self._start_time
            rules = self._get_current_optimized_rules(xk)
            
            if rules:
                print(f"DE Iter {self._iteration_count}: Best MSE: {self._last_best_score:.6f} - Time: {elapsed_time:.1f}s")
                print("  Current optimized rules:")
                for rule in rules:
                    print(f"    {rule}")
                print()  # Add blank line for readability
        elif self.verbose:
            # Verbose mode: show iteration details only when not displaying rules
            print(f"Iteration {self._iteration_count}: Current MSE: {current_score:.6f}, Best MSE so far: {self._last_best_score:.6f}, Convergence: {convergence:.4f}")
        elif not should_display_rules:
            # Non-verbose mode: show progress bar
            sys.stdout.write(f'\r[{bar}] {progress:.1f}% | Iter: {self._iteration_count}/{self._max_iter} | Best MSE: {self._last_best_score:.6f} | Conv: {convergence:.4f}')
            sys.stdout.flush()

        # Early stopping logic
        de_params = self.config.get('de_params', {})
        early_stopping = de_params.get('early_stopping', False)
        if early_stopping:
            success_threshold = self.config.get('success_threshold', 0.01)
            if current_score < success_threshold:
                if self.verbose or should_display_rules: 
                    print(f"\nDE early stopping: MSE below success threshold: {success_threshold}")
                else:
                    # Clear progress line before printing early stop message
                    sys.stdout.write('\r' + ' ' * 120 + '\r')
                    print(f"DE early stopping: MSE below success threshold: {success_threshold}")
                return True

    def _get_current_optimized_rules(self, current_params: np.ndarray) -> List[str]:
        """
        Get formatted rules for nodes being optimized with current parameter values.
        
        Parameters:
        -----------
        current_params : np.ndarray
            Current parameter vector from optimization
            
        Returns:
        --------
        List[str]
            List of formatted rule strings for nodes being optimized
        """
        try:
            # Convert current parameters to Cij matrix
            current_cij = self.evaluator._vector_to_cij_matrix(current_params)
            
            # Get only the nodes being optimized
            rules = []
            for node_name in self.evaluator.nodes_to_optimize:
                if (node_name in self.pbn.nodeDict and 
                    hasattr(self.pbn, 'gene_functions') and 
                    node_name in self.pbn.gene_functions):
                    
                    node_idx = self.pbn.nodeDict[node_name]
                    functions = self.pbn.gene_functions[node_name]
                    
                    # Get the actual number of functions for this node
                    num_functions = len(functions)
                    if num_functions == 0:
                        continue
                    
                    # Get probabilities for this node, ensuring we don't exceed bounds
                    if node_idx < current_cij.shape[0] and num_functions <= current_cij.shape[1]:
                        probabilities = current_cij[node_idx, :num_functions]
                        
                        # Normalize probabilities to ensure they sum to 1
                        prob_sum = np.sum(probabilities)
                        if prob_sum > 1e-10:  # Avoid division by zero
                            probabilities = probabilities / prob_sum
                        
                        # Format the rule string
                        rule_parts = []
                        for i, (func, prob) in enumerate(zip(functions, probabilities)):
                            if prob > 1e-6:  # Only show functions with non-negligible probability
                                rule_parts.append(f"{func} (p={prob:.3f})")
                        
                        if rule_parts:
                            rules.append(f"{node_name}: {' | '.join(rule_parts)}")
            
            return rules
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not generate rules: {str(e)}")
            return []

    def _pso_optimizer(self, bounds):
        """PSO implementation using pyswarms library"""
        pso_params = self.config['pso_params'].copy()
        
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        pyswarms_bounds = (lb, ub)
        
        # Extract parameters
        n_particles = pso_params.get('n_particles', 30)
        iters = pso_params.get('iters', 100)
        ftol = pso_params.get('ftol', 1e-3)
        ftol_iter = pso_params.get('ftol_iter', 10)

        # Print optimization setup info 
        print(f"Running PSO optimization...")
        print("\n")
        if self.verbose:
            print(f"PSO Setup:")
            print(f"  - Particles: {n_particles}")
            print(f"  - Max iterations: {iters}")
            print(f"  - Problem dimensions: {len(bounds)}")
            print(f"  - Total function evaluations: {n_particles * iters}")
            if ftol > 0:
                print(f"  Early stopping enabled:")
                print(f"    - Function tolerance: {ftol}")
                print(f"    - Tolerance iterations: {ftol_iter}")

        def pso_objective(x):
            """Objective function wrapper for PSO"""
            costs = []
            for i, p in enumerate(x):
                cost = self.evaluator.objective_function(p)
                costs.append(cost)
            return np.array(costs)

        # Create optimizer
        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=len(bounds),
            options=pso_params.get('options', {'c1': 0.5, 'c2': 0.3, 'w': 0.9}),
            bounds=pyswarms_bounds,
            ftol=ftol,
            ftol_iter=ftol_iter
        )
        
        # Run optimization
        cost, pos = optimizer.optimize(pso_objective, iters=iters)

        # Check if early stopping occurred and determine success
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
                                 show_stagnation: bool = False, log_scale: bool = False, sorted: bool = False):
        """
        Plot the optimization history (MSE over iterations).
        
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
        sorted: bool, optional
            Whether to sort the optimization history by MSE
        """
        import matplotlib.pyplot as plt
        
        if not hasattr(result, 'history') or not result.history:
            print("No optimization history available for plotting.")
            return
        
        history = np.array(result.history)
        if sorted:
            history = np.sort(history)
        plt.figure(figsize=(10, 6))
        plt.plot(history, 'b-', linewidth=2, label='MSE')
        
        if log_scale:
            plt.yscale('log')
        
        if show_stagnation:
            stagnation_periods = self._detect_stagnation_periods(history)
            for start, end in stagnation_periods:
                plt.axvspan(start, end, alpha=0.3, color='red', label='Stagnation' if start == stagnation_periods[0][0] else "")
        
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title(f'Optimization History')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add text box with final statistics
        stats_text = f'Final MSE: {result.fun:.6f}\nIterations: {result.nit}'
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
        optimized_pbn = self.pbn
        
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

    def test_steady_states(self, test_experiments=None, show_plot=False, convergence_threshold=1.0):
        """
        Test steady state calculation methods for each experiment condition.
        Uses the optimizer's existing configuration and reports timing and convergence.
        
        Parameters:
        -----------
        test_experiments : list, optional
            List of experiment dictionaries to test. If None, uses self.evaluator.experiments
        show_plot : bool, default=False
            Whether to display convergence plots for each experiment
        convergence_threshold : float, default=1.0
            Threshold for considering convergence successful (in percentage)
        """
        print("=== Steady State Methods Testing ===")
        
        experiments = test_experiments or self.evaluator.experiments
        
        if not experiments:
            print("No experiments available for testing.")
            return
            
        # Get steady state config from optimizer
        ss_config = self.config.get('steady_state', self._default_config()['steady_state'])
        method = ss_config.get('method', 'tsmc')
        
        print(f"Testing {len(experiments)} experimental conditions using {method.upper()} method...")
        print(f"Steady state config: {ss_config}")
        if method == 'monte_carlo':
            print(f"Convergence threshold: {convergence_threshold}%")
        print()
        
        results = []
        converged_experiments = 0
        
        for i, experiment in enumerate(experiments):
            print(f"Experiment {i+1}: ", end="")
            
            # Display experimental conditions briefly
            stimuli_str = ",".join(experiment['stimuli']) if experiment['stimuli'] else "None"
            inhibitors_str = ",".join(experiment['inhibitors']) if experiment['inhibitors'] else "None"
            print(f"Stimuli={stimuli_str}, Inhibitors={inhibitors_str}")
            
            start_time = time.time()
            success = False
            converged = False
            
            try:
                # Set experimental conditions
                self.evaluator.steady_state_calc.set_experimental_conditions(
                    stimuli=experiment['stimuli'],
                    stimuli_efficacy=experiment['stimuli_efficacy'],
                    inhibitors=experiment['inhibitors'],
                    inhibitors_efficacy=experiment['inhibitors_efficacy']
                )
                
                # Run steady state calculation using optimizer's config
                if method == 'tsmc':
                    params = ss_config.get('tsmc_params', {})
                    result = self.evaluator.steady_state_calc.compute_stationary_tsmc(**params)
                else:  # monte_carlo
                    params = ss_config.get('monte_carlo_params', {})
                    
                    # Use convergence analysis for Monte Carlo
                    params_with_convergence = params.copy()
                    params_with_convergence['analyze_convergence'] = True
                    params_with_convergence['show_plot'] = show_plot
                    
                    # Pick a measured node for convergence analysis if available
                    output_node = None
                    if experiment['measurements']:
                        output_node = list(experiment['measurements'].keys())[0]
                        params_with_convergence['output_node'] = output_node
                    
                    result, convergence_info = self.evaluator.steady_state_calc.compute_stationary_mc(**params_with_convergence)
                    
                    # Check convergence success
                    final_change = convergence_info['final_relative_change']
                    converged = final_change <= convergence_threshold
                    if converged:
                        converged_experiments += 1
                
                elapsed_time = time.time() - start_time
                success = True
                
                # Print results
                if method == 'monte_carlo':
                    convergence_status = "✓ Converged" if converged else "✗ Not converged"
                    print(f"  Time: {elapsed_time:.3f}s, Final change: {final_change:.2f}%, {convergence_status}")
                else:
                    print(f"  Time: {elapsed_time:.3f}s")
                
                # Reset conditions
                self.evaluator.steady_state_calc.reset_network_conditions()
                
                results.append({
                    'experiment_id': experiment.get('id', i+1),
                    'success': success,
                    'time': elapsed_time,
                    'method': method,
                    'converged': converged if method == 'monte_carlo' else None,
                    'final_change': final_change if method == 'monte_carlo' else None
                })
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"  FAILED after {elapsed_time:.3f}s: {str(e)}")
                results.append({
                    'experiment_id': experiment.get('id', i+1),
                    'success': False,
                    'time': elapsed_time,
                    'method': method,
                    'error': str(e)
                })
        
        # Summary
        successes = sum(1 for r in results if r['success'])
        total_time = sum(r['time'] for r in results)
        
        print(f"\nSummary: {successes}/{len(results)} experiments successful, total time: {total_time:.3f}s")
        
        if method == 'monte_carlo':
            print(f"Convergence: {converged_experiments}/{len(results)} experiments converged (threshold: {convergence_threshold}%)")
        
        return results 