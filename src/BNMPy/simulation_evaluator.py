import numpy as np
import itertools
from typing import Dict, List, Union, Optional
from BNMPy.steady_state import SteadyStateCalculator

class SimulationEvaluator:
    """
    Evaluation engine for PBN parameter optimization
    Handles experiment simulation and SSE calculation
    """
    
    def __init__(self, pbn, experiments, config=None, nodes_to_optimize=None, normalize=False):
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
        normalize : bool, default=False
            Whether to normalize formula-based measurements using min-max scaling across experiments
        """
        self.pbn = pbn
        self.experiments = experiments
        self.steady_state_calc = SteadyStateCalculator(pbn)
        self.config = config or self._default_config()
        self.normalize = normalize
        
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
        
        # Pre-compute normalized measured values for formula-based experiments if normalize=True
        self.normalized_measured_values = {}
        if self.normalize:
            self._normalize_measured_values()

        # Check formula-based measurements and warn if measured values are outside theoretical range
        # Skip warning if normalize=True since values will be normalized anyway
        if not self.normalize:
            formula_experiments = [exp for exp in self.experiments if exp.get('measured_formula')]
            if formula_experiments:
                formula = formula_experiments[0]['measured_formula']
                variables = self._extract_variables_from_formula(formula)
                
                try:
                    # Calculate theoretical range by evaluating formula at all corners
                    n_vars = len(variables)
                    formula_values = []
                    
                    for combo in itertools.product([0.0, 1.0], repeat=n_vars):
                        var_values = dict(zip(variables, combo))
                        for node in self.pbn.nodeDict.keys():
                            if node not in var_values:
                                var_values[node] = 0.0
                        value = self._safe_eval_formula(formula, var_values)
                        formula_values.append(value)
                    
                    theoretical_min = float(np.min(formula_values))
                    theoretical_max = float(np.max(formula_values))
                    
                    # Check if any measured values are outside theoretical range
                    for exp in formula_experiments:
                        mv = exp.get('measured_value')
                        if mv is not None and (mv < theoretical_min or mv > theoretical_max):
                            print(f"WARNING: Experiment {exp['id']} measured value {mv} is outside theoretical range [{theoretical_min}, {theoretical_max}]")
                            print(f"         Formula: {formula}")
                            print(f"         Consider rescaling your measured values to match the formula's range.")
                except Exception as e:
                    print(f"Warning: Could not validate formula range: {e}")
        
    def _normalize_measured_values(self):
        """
        Pre-compute min-max normalized measured values for all experiments.
        Stores mapping from (experiment id, node name) to normalized measured value.
        """
        # Collect all measured values across all experiments
        all_measured_values = []
        
        # For formula-based experiments
        for exp in self.experiments:
            if exp.get('measured_formula'):
                mv = exp.get('measured_value')
                if mv is not None:
                    all_measured_values.append(float(mv))
            else:
                # For node-based measurements
                for node_name, measured_value in exp['measurements'].items():
                    all_measured_values.append(float(measured_value))
        
        if len(all_measured_values) == 0:
            return
        
        # Calculate min and max across ALL measured values
        min_val = min(all_measured_values)
        max_val = max(all_measured_values)
        
        # Store the range for later use
        self.measured_value_range = (min_val, max_val)
        
        # Normalize each experiment's measured values
        for exp in self.experiments:
            if exp.get('measured_formula'):
                # Formula-based measurement
                mv = exp.get('measured_value')
                if mv is not None:
                    if max_val - min_val > 1e-10:  # Avoid division by zero
                        normalized = (float(mv) - min_val) / (max_val - min_val)
                    else:
                        normalized = 0.5  # If all values are the same, use 0.5
                    self.normalized_measured_values[exp['id']] = normalized
            else:
                # Node-based measurements
                for node_name, measured_value in exp['measurements'].items():
                    if max_val - min_val > 1e-10:
                        normalized = (float(measured_value) - min_val) / (max_val - min_val)
                    else:
                        normalized = 0.5
                    # Use tuple (exp_id, node_name) as key for node-based measurements
                    self.normalized_measured_values[(exp['id'], node_name)] = normalized
        
        print(f"Normalization enabled: Measured values range [{min_val:.4f}, {max_val:.4f}] scaled to [0, 1]")
    
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
                    'freeze_constant': True
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
            
            # Check measured nodes or formula
            if exp.get('measured_formula'):
                # Validate that variables in formula exist in node dict
                vars_in_formula = self._extract_variables_from_formula(exp['measured_formula'])
                for var in vars_in_formula:
                    if var not in node_dict:
                        raise ValueError(f"Variable {var} in measured formula not found in network")
            else:
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
        # If normalize=True, collect all predicted values first for normalization
        max_retries = 3
        
        if self.normalize:
            # Collect all predicted values across experiments first
            experiment_predictions = []
            
            for i, experiment in enumerate(self.experiments):
                success = False
                for retry in range(max_retries):
                    try:
                        predicted = self._simulate_experiment(experiment)
                        
                        if np.any(np.isnan(predicted)) or np.any(np.isinf(predicted)):
                            raise ValueError("Invalid simulation results (NaN or Inf)")
                        
                        # Store the predicted steady state for later use
                        experiment_predictions.append({
                            'experiment': experiment,
                            'predicted': predicted
                        })
                        success = True
                        break
                    except Exception as e:
                        print(f"  Warning: Simulation retry {retry+1}/{max_retries} failed: {str(e)}")
                        if retry == max_retries - 1:
                            print(f"  Objective function penalty: Experiment simulation failed after {max_retries} retries.")
                            return 1e10
                        self.pbn.resetNetwork()
                
                if not success:
                    print("  Objective function penalty: Experiment simulation did not succeed.")
                    return 1e10
            
            # Calculate normalized SSE
            total_sse = self._calculate_normalized_sse(experiment_predictions)
        else:
            # Standard SSE calculation without normalization
            total_sse = 0
            
            for i, experiment in enumerate(self.experiments):
                success = False
                for retry in range(max_retries):
                    try:
                        predicted = self._simulate_experiment(experiment)
                        
                        if np.any(np.isnan(predicted)) or np.any(np.isinf(predicted)):
                            raise ValueError("Invalid simulation results (NaN or Inf)")
                        
                        sse = self._calculate_sse(predicted, experiment)
                        
                        if np.isfinite(sse):
                            total_sse += sse
                            success = True
                            break
                    except Exception as e:
                        print(f"  Warning: Simulation retry {retry+1}/{max_retries} failed: {str(e)}")
                        if retry == max_retries - 1:
                            print(f"  Objective function penalty: Experiment simulation failed after {max_retries} retries.")
                            return 1e10
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
            params['seed'] = self.config.get('seed', 9)
            # print(f"  TSMC params: {params}")
            steady_state = self.steady_state_calc.compute_stationary_tsmc(**params)
        else:  # monte_carlo
            params = ss_config.get('monte_carlo_params', {})
            params['seed'] = self.config.get('seed', 9)
            # print(f"  Monte Carlo params: {params}")
            steady_state = self.steady_state_calc.compute_stationary_mc(**params)
        
        # 3. Reset experimental conditions
        self.steady_state_calc.reset_network_conditions()
        
        return steady_state
    
    def _calculate_normalized_sse(self, experiment_predictions: List[Dict]) -> float:
        """
        Calculate SSE with min-max normalization of predicted values across all experiments.
        
        Parameters:
        -----------
        experiment_predictions : List[Dict]
            List of dicts containing 'experiment' and 'predicted' keys for each experiment
            
        Returns:
        --------
        float
            Sum of squared errors with normalized values
        """
        # Collect all predicted values (both formula and node-based)
        all_predicted_values = []
        predicted_info = []  # Store (type, value, experiment, predicted_array)
        
        for pred_info in experiment_predictions:
            experiment = pred_info['experiment']
            predicted = pred_info['predicted']
            
            if experiment.get('measured_formula'):
                # Formula-based measurement
                var_values = {name: predicted[idx] for name, idx in self.pbn.nodeDict.items()}
                predicted_value = self._safe_eval_formula(experiment['measured_formula'], var_values)
                all_predicted_values.append(predicted_value)
                predicted_info.append(('formula', predicted_value, experiment, predicted))
            else:
                # Node-based measurements
                for node_name, measured_value in experiment['measurements'].items():
                    node_idx = self.pbn.nodeDict[node_name]
                    predicted_value = predicted[node_idx]
                    all_predicted_values.append(predicted_value)
                    predicted_info.append(('node', predicted_value, experiment, predicted, node_name))
        
        # Calculate min-max range for all predicted values
        if len(all_predicted_values) > 0:
            pred_min = min(all_predicted_values)
            pred_max = max(all_predicted_values)
        else:
            pred_min = 0.0
            pred_max = 1.0
        
        # Calculate SSE with normalized values
        total_sse = 0.0
        for info in predicted_info:
            if info[0] == 'formula':
                # Formula-based measurement
                _, predicted_value, experiment, _ = info
                
                # Normalize predicted value
                if pred_max - pred_min > 1e-10:
                    normalized_predicted = (predicted_value - pred_min) / (pred_max - pred_min)
                else:
                    normalized_predicted = 0.5
                
                # Get normalized measured value
                normalized_measured = self.normalized_measured_values.get(experiment['id'], 0.5)
                
                # Calculate squared error
                sse = (normalized_predicted - normalized_measured) ** 2
                total_sse += sse
            else:
                # Node-based measurement
                _, predicted_value, experiment, _, node_name = info
                
                # Normalize predicted value
                if pred_max - pred_min > 1e-10:
                    normalized_predicted = (predicted_value - pred_min) / (pred_max - pred_min)
                else:
                    normalized_predicted = 0.5
                
                # Get normalized measured value
                normalized_measured = self.normalized_measured_values.get((experiment['id'], node_name), 0.5)
                
                # Calculate squared error
                sse = (normalized_predicted - normalized_measured) ** 2
                total_sse += sse
        
        return total_sse
    
    def _calculate_sse(self, predicted: np.ndarray, experiment: Dict) -> float:
        """
        Calculate sum of squared errors for measured nodes
        
        Parameters:
        -----------
        predicted : np.ndarray
            Predicted steady state probabilities
        experiment : dict
            Experiment dictionary containing either node measurements or a measured formula
            
        Returns:
        --------
        float
            Sum of squared errors
        """
        # Formula-based single measurement
        if experiment.get('measured_formula'):
            measured_value = float(experiment.get('measured_value')) if experiment.get('measured_value') is not None else 0.0
            
            # Compute predicted formula value
            var_values = {name: predicted[idx] for name, idx in self.pbn.nodeDict.items()}
            predicted_value = self._safe_eval_formula(experiment['measured_formula'], var_values)
            # print(f"  Predicted formula value: {predicted_value}, Measured value: {measured_value}")
            
            return float(predicted_value - measured_value) ** 2
        
        # Standard per-node SSE
        sse = 0.0
        for node, measured_value in experiment['measurements'].items():
            node_idx = self.pbn.nodeDict[node]
            predicted_value = predicted[node_idx]
            sse += (predicted_value - measured_value) ** 2
        return sse

    @staticmethod
    def _extract_variables_from_formula(formula: str):
        import re
        tokens = re.findall(r"\b[A-Za-z_]\w*\b", str(formula))
        return list(dict.fromkeys(tokens))

    @staticmethod
    def _safe_eval_formula(formula: str, variables: Dict[str, float]) -> float:
        import ast
        import operator as op

        allowed_operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.USub: op.neg,
            ast.UAdd: op.pos,
            ast.Pow: op.pow,
        }

        def _eval(node):
            if isinstance(node, ast.Num):  # type: ignore[attr-defined]
                return node.n
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError("Invalid constant in formula")
            if isinstance(node, ast.Name):
                if node.id in variables:
                    return variables[node.id]
                raise ValueError(f"Unknown variable '{node.id}' in formula")
            if isinstance(node, ast.BinOp):
                if type(node.op) not in allowed_operators:
                    raise ValueError("Operator not allowed in formula")
                return allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp):
                if type(node.op) not in allowed_operators:
                    raise ValueError("Unary operator not allowed in formula")
                return allowed_operators[type(node.op)](_eval(node.operand))
            raise ValueError("Unsupported expression in formula")

        tree = ast.parse(str(formula), mode='eval')
        return float(_eval(tree.body))
    
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