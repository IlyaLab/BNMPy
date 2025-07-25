import numpy as np
from typing import Dict, List, Union, Optional
from BNMPy.steady_state import SteadyStateCalculator

class SimulationEvaluator:
    """
    Evaluation engine for PBN parameter optimization
    Handles experiment simulation and SSE calculation
    """
    
    def __init__(self, pbn, experiments, config=None, nodes_to_optimize=None):
        """
        Initialize evaluator with PBN and experiments
        
        Parameters:
        -----------
        pbn : ProbabilisticBN
            BNMPy PBN object
        experiments : list
            List of experiment dictionaries from ExperimentData
        config : dict, optional
            Configuration parameters for simulation
        nodes_to_optimize : list, optional
            List of node names to optimize. If None, optimize all nodes.
        """
        self.pbn = pbn
        self.experiments = experiments
        self.steady_state_calc = SteadyStateCalculator(pbn)
        self.config = config or self._default_config()
        
        # Set up nodes to optimize
        self.nodes_to_optimize = nodes_to_optimize
        if nodes_to_optimize is not None:
            # Validate nodes exist in network
            for node in nodes_to_optimize:
                if node not in self.pbn.nodeDict:
                    raise ValueError(f"Node {node} not found in network")
            self.node_indices_to_optimize = [self.pbn.nodeDict[node] for node in nodes_to_optimize]
        else:
            self.node_indices_to_optimize = list(range(len(self.pbn.nf)))
        
        # Validate experiments against network structure
        self._validate_experiments()
        
        # Store original parameters for non-optimized nodes
        self._store_original_parameters()
        
    def _store_original_parameters(self):
        """Store original parameters for non-optimized nodes"""
        self.original_cij = self.pbn.cij.copy()
        
    def _default_config(self) -> Dict:
        """Default simulation configuration"""
        return {
            'steady_state': {
                'method': 'monte_carlo',  # or 'tsmc'
                'tsmc_params': {
                    'epsilon': 0.001,
                    'r': 0.025,
                    's': 0.95,
                    'p_mir': 0.001,
                    'initial_nsteps': 100,
                    'max_iterations': 500,
                    'freeze_self_loop': True
                },
                'monte_carlo_params': {
                    'n_runs': 3,
                    'n_steps': 2000,
                    'p_noise': 0.05
                }
            }
        }
        
    def _validate_experiments(self):
        """Validate experiments against network structure"""
        node_dict = self.pbn.nodeDict
        for exp in self.experiments:
            # Check stimuli nodes
            for node in exp['stimuli']:
                if node not in node_dict:
                    raise ValueError(f"Stimulus node {node} not found in network")
            
            # Check inhibitor nodes
            for node in exp['inhibitors']:
                if node not in node_dict:
                    raise ValueError(f"Inhibitor node {node} not found in network")
            
            # Check measured nodes from the measurements dictionary keys
            for node in exp['measurements'].keys():
                if node not in node_dict:
                    raise ValueError(f"Measured node {node} not found in network")
    
    def objective_function(self, cij_vector: np.ndarray) -> float:
        """
        Calculate objective function (MSE) for given parameters
        
        Parameters:
        -----------
        cij_vector : np.ndarray
            Flattened selection probability vector for optimized nodes
            
        Returns:
        --------
        float
            Mean squared error across all experiments
        """
        # print(f"\n--- Evaluating objective function ---")
        # print(f"Received cij_vector: {cij_vector}")

        # 1. Handle numerical issues in input vector
        if np.any(np.isnan(cij_vector)) or np.any(np.isinf(cij_vector)):
            # print("Objective function penalty: NaN or Inf in input vector.")
            return 1e10  # Large but finite penalty
            
        # 2. Reshape vector to Cij matrix and validate
        try:
            cij_matrix = self._vector_to_cij_matrix(cij_vector)
            # print(f"Reshaped Cij matrix:\n{cij_matrix}")
            if not self._validate_cij_matrix(cij_matrix, verbose=False):
                # print("Objective function penalty: Invalid Cij matrix.")
                return 1e10  # Large but finite penalty
        except Exception as e:
            print(f"Warning: Error in parameter conversion: {str(e)}")
            return 1e10
        
        # 3. Update PBN with new parameters
        try:
            self._update_pbn_parameters(cij_matrix)
        except Exception as e:
            print(f"Warning: Error updating PBN parameters: {str(e)}")
            return 1e10
        
        # 4. Calculate SSE across all experiments
        total_sse = 0
        max_retries = 3
        
        for i, experiment in enumerate(self.experiments):
            # print(f"\n- Simulating experiment {i+1} -")
            success = False
            for retry in range(max_retries):
                try:
                    predicted = self._simulate_experiment(experiment)
                    # print(f"  Predicted steady state: {predicted}")

                    if np.any(np.isnan(predicted)) or np.any(np.isinf(predicted)):
                        raise ValueError("Invalid simulation results (NaN or Inf)")
                    
                    sse = self._calculate_sse(predicted, experiment['measurements'])
                    # print(f"  SSE for experiment {i+1}: {sse}")

                    if np.isfinite(sse):
                        total_sse += sse
                        success = True
                        break
                except Exception as e:
                    print(f"  Warning: Simulation retry {retry+1}/{max_retries} failed: {str(e)}")
                    if retry == max_retries - 1:
                        print(f"  Objective function penalty: Experiment simulation failed after {max_retries} retries.")
                        return 1e10
                    # Reset network state and try again
                    self.pbn.resetNetwork()
            
            if not success:
                print("  Objective function penalty: Experiment simulation did not succeed.")
                return 1e10
        
        # 5. Calculate MSE by dividing by number of experiments
        mse = total_sse / len(self.experiments)
        
        # # 6. Add small regularization term to prevent degenerate solutions
        # regularization = 1e-6 * np.sum(np.square(cij_vector))
        # mse += regularization
        
        # print(f"\nTotal MSE: {mse}")
        # print(f"--- Finished objective function evaluation ---\n")
        return mse
    
    def _vector_to_cij_matrix(self, cij_vector: np.ndarray) -> np.ndarray:
        """
        Convert flattened parameter vector to Cij matrix
        handles selective node optimization
        
        Parameters:
        -----------
        cij_vector : np.ndarray
            Flattened selection probability vector for optimized nodes
            
        Returns:
        --------
        np.ndarray
            Complete Cij matrix with optimized and fixed parameters
        """
        nf = self.pbn.nf  # Number of functions per node
        max_funcs = np.max(nf)  # Maximum number of functions for any node
        cij_matrix = self.original_cij.copy()  # Start with original parameters
        
        # Convert vector parameters only for nodes being optimized
        vector_idx = 0
        for node_idx in self.node_indices_to_optimize:
            n_funcs = nf[node_idx]
            node_probs = cij_vector[vector_idx:vector_idx + n_funcs].copy()
            
            # Handle edge cases and normalize probabilities
            prob_sum = np.sum(node_probs)
            
            if prob_sum <= 1e-9 or not np.isfinite(prob_sum):
                # If sum is effectively zero or invalid, use uniform distribution
                node_probs = np.ones(n_funcs) / n_funcs
            elif not np.isclose(prob_sum, 1.0, rtol=1e-5):
                # Normalize to sum to 1
                node_probs = node_probs / prob_sum
            
            # Ensure probabilities are within valid bounds
            node_probs = np.clip(node_probs, 1e-10, 1.0)
            
            # Renormalize after clipping to ensure they still sum to 1
            node_probs = node_probs / np.sum(node_probs)
            
            # Create a new, clean row for the optimized node, padded with -1
            new_row = np.full(max_funcs, -1.0)
            new_row[:n_funcs] = node_probs
            cij_matrix[node_idx, :] = new_row
            
            vector_idx += n_funcs
            
        return cij_matrix
    
    def _validate_cij_matrix(self, cij_matrix: np.ndarray, verbose: bool = False) -> bool:
        """
        Validate Cij matrix constraints with numerical tolerance
        
        Parameters:
        -----------
        cij_matrix : np.ndarray
            Selection probability matrix
        verbose : bool, optional
            Whether to print validation error messages (default: False during optimization)
            
        Returns:
        --------
        bool
            True if matrix satisfies all constraints
        """
        # Check for valid probability values, ignoring -1 placeholders
        if np.any((cij_matrix < -1e-10) & (cij_matrix != -1)):
            if verbose:
                print(f"Validation failed: Cij matrix contains invalid negative values.")
            return False
        if np.any(cij_matrix > 1 + 1e-10):
            if verbose:
                print(f"Validation failed: Cij matrix contains values greater than 1.")
            return False
            
        # Check that rows for optimized nodes sum to 1
        temp_cij = cij_matrix.copy()
        temp_cij[temp_cij == -1] = 0 # Treat placeholders as 0 for summation

        for node_idx in self.node_indices_to_optimize:
            row_sum = np.sum(temp_cij[node_idx, :])
            if not np.isclose(row_sum, 1.0, rtol=1e-5, atol=1e-8):
                if verbose:
                    node_name = list(self.pbn.nodeDict.keys())[node_idx]
                    print(f"Validation failed for optimized node '{node_name}': Probabilities do not sum to 1 (sum={row_sum}).")
                return False
            
        return True
    
    def _update_pbn_parameters(self, cij_matrix: np.ndarray):
        """
        Update PBN with new selection probabilities
        
        Parameters:
        -----------
        cij_matrix : np.ndarray
            Selection probability matrix
        """
        self.pbn.cij = cij_matrix
        # Update cumulative probabilities if needed
        if hasattr(self.pbn, 'update_cumulative_probabilities'):
            self.pbn.update_cumulative_probabilities()
    
    def _simulate_experiment(self, experiment: Dict) -> np.ndarray:
        """
        Simulate single experiment to steady state
        
        Parameters:
        -----------
        experiment : dict
            Experiment configuration
            
        Returns:
        --------
        np.ndarray
            Steady state probabilities
        """
        # 1. Set experimental conditions with efficacy values
        self.steady_state_calc.set_experimental_conditions(
            stimuli=experiment['stimuli'],
            stimuli_efficacy=experiment['stimuli_efficacy'],
            inhibitors=experiment['inhibitors'],
            inhibitors_efficacy=experiment['inhibitors_efficacy']
        )
        # print(f"  Simulating with Stimuli: {experiment['stimuli']} (efficacy: {experiment['stimuli_efficacy']}), Inhibitors: {experiment['inhibitors']} (efficacy: {experiment['inhibitors_efficacy']})")

        # 2. Calculate steady state
        ss_config = self.config.get('steady_state', self._default_config()['steady_state'])
        method = ss_config.get('method', 'tsmc')
        
        # print(f"  Using steady-state method: {method}")
        if method == 'tsmc':
            params = ss_config.get('tsmc_params', {})
            # print(f"  TSMC params: {params}")
            steady_state = self.steady_state_calc.compute_stationary_tsmc(**params)
        else:  # monte_carlo
            params = ss_config.get('monte_carlo_params', {})
            # print(f"  Monte Carlo params: {params}")
            steady_state = self.steady_state_calc.compute_stationary_mc(**params)
        
        # 3. Reset experimental conditions
        self.steady_state_calc.reset_network_conditions()
        
        return steady_state
    
    def _calculate_sse(self, predicted: np.ndarray, measurements: Dict[str, float]) -> float:
        """
        Calculate sum of squared errors for measured nodes
        
        Parameters:
        -----------
        predicted : np.ndarray
            Predicted steady state probabilities
        measurements : dict
            Measured values for nodes
            
        Returns:
        --------
        float
            Sum of squared errors
        """
        sse = 0.0
        for node, measured_value in measurements.items():
            node_idx = self.pbn.nodeDict[node]
            predicted_value = predicted[node_idx]
            sse += (predicted_value - measured_value) ** 2
        return sse
    
    def get_parameter_bounds(self) -> List[tuple]:
        """
        Get bounds for optimization parameters
        Now returns bounds only for nodes being optimized
        
        Returns:
        --------
        list
            List of (min, max) tuples for each parameter
        """
        bounds = []
        for node_idx in self.node_indices_to_optimize:
            # Each parameter between 0 and 1
            bounds.extend([(0, 1) for _ in range(self.pbn.nf[node_idx])])
        return bounds 