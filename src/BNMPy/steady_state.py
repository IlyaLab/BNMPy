import numpy as np
import warnings
from typing import Union, Dict, List, Tuple, Optional, Any


class SteadyStateCalculator:
    """
    Steady-state calculation for Boolean and Probabilistic Boolean Network
    Supports both Two-State Markov Chain (TSMC) and Monte Carlo methods
    Reference: pbnStationary_TS.m from optPBN (Trairatphisan et al. 2014)
    """
    
    def __init__(self, network):
        self.network = network
        self.N = network.N  # Number of nodes
        
        # Extract network properties based on type
        if hasattr(network, 'cij'):  # ProbabilisticBN
            self.is_pbn = True
            self.varF = network.varF
            self.nf = network.nf
            self.F = network.F
            self.cij = network.cij
            self.K = network.K
            self.cumsum = network.cumsum
            self.Nf = network.Nf
        else:  # BooleanNetwork
            self.is_pbn = False
            self.varF = network.varF
            self.F = network.F
            self.K = network.K
            # Convert to PBN-like format for unified processing
            self.nf = np.ones(self.N, dtype=int)
            self.cij = np.ones((self.N, 1))
            self.cumsum = np.arange(self.N + 1)
            self.Nf = self.N
            
        # Identify input nodes (constant or self-dependent only)
        self.input_indices = self._identify_input_nodes()
        
        # Store original network state for restoration
        self._original_nodes = None
        
    def _identify_input_nodes(self) -> List[int]:
        """
        Identify input nodes:
        - sum(nf*nv) == 0 (no inputs) AND not a knockdown node
        - sum(nf*nv) == 1 and self-dependent (self-dependent only) AND not a knockdown node

        These will not be perturbed.
        """
        input_indices = []
        running_index = 0
        
        for i in range(len(self.nf)):
            nfnv = []
            for j in range(self.nf[i]): # for each function/node
                # Calculate nf[i] * nv[j] equivalent
                func_idx = running_index + j
                if func_idx < len(self.K):
                    nfnv.append(self.nf[i] * self.K[func_idx])
                else:
                    nfnv.append(0)
            
            nfnv_sum = sum(nfnv)
            
            # Check if this is a knockdown node
            is_knockdown = hasattr(self.network, 'knockdown_nodes') and i in self.network.knockdown_nodes
            
            # Check if input: sum=0 (no inputs) or sum=1 and self-dependent
            # exclude knockdown nodes
            if not is_knockdown and (nfnv_sum == 0 or (nfnv_sum == 1 and self._is_self_dependent(i))):
                input_indices.append(i)
                
            running_index += self.nf[i]
            
        return input_indices
    
    def _is_self_dependent(self, node_idx: int) -> bool:
        """Check if a node depends only on itself"""
        if self.is_pbn:
            func_start = self.cumsum[node_idx]
            func_end = self.cumsum[node_idx + 1]
            for func_idx in range(func_start, func_end):
                if (self.K[func_idx] == 1 and 
                    self.varF[func_idx, 0] == node_idx):
                    return True
        else:
            if (self.K[node_idx] == 1 and 
                self.varF[node_idx, 0] == node_idx):
                return True
        return False
    
    def _get_node_name_by_index(self, node_idx: int) -> Optional[str]:
        """Get node name by index from nodeDict"""
        for name, idx in self.network.nodeDict.items():
            if idx == node_idx:
                return name
        return None
    
    def _set_input_nodes_to_zero(self):
        """Set all input nodes to 0 by default using knockout"""
        for i in self.input_indices:
            node_name = self._get_node_name_by_index(i)
            if node_name is not None:
                self.network.knockout(node_name, 0)
    
    def compute_steady_state(self, method: str = 'tsmc', **kwargs) -> np.ndarray:
        """
        steady-state calculation
        If BN, run deterministic steady-state calculation.
        If PBN, run Two-State Markov Chain or Monte Carlo.
        
        Parameters:
        -----------
        method : str
            'tsmc' for Two-State Markov Chain or 'monte_carlo' for Monte Carlo
        **kwargs : dict
            Method-specific parameters
            
        Returns:
        --------
        np.ndarray : Steady-state probabilities for each node
        """
        if not self.is_pbn:
            return self.compute_stationary_deterministic(**kwargs)
        
        if method == 'tsmc':
            return self.compute_stationary_tsmc(**kwargs)
        elif method == 'monte_carlo':
            return self.compute_stationary_mc(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tsmc' or 'monte_carlo'")
    
    def compute_stationary_tsmc(self, 
                               epsilon: float = 0.001,
                               r: float = 0.025, 
                               s: float = 0.95,
                               p_noise: float = 0,
                               p_mir: float = 0.001,
                               initial_nsteps: int = 100,
                               max_iterations: int = 500,
                               freeze_constant: bool = False,
                               verbose: bool = False,
                               seed: Optional[int] = None) -> np.ndarray:
        """
        Two-State Markov Chain steady-state calculation
        In addition to the original function, handle input nodes and constant nodes
        
        Ref: pbnStationary_TS.m Approach 1

        Parameters:
        -----------
        - `epsilon` (float, default=0.001): Range of transition probability (smaller = more accurate)
        - `r` (float, default=0.025): Range of accuracy - most sensitive parameter (smaller = more accurate)
        - `s` (float, default=0.95): Probability of accuracy (closer to 1 = more confident)
        - `p_noise` (float, default=0): Noise probability for Monte Carlo method
        - `p_mir` (float, default=0.001): Perturbation probability (Miranda-Parga scheme)
        - `initial_nsteps` (int, default=100): Initial number of simulation steps
        - `max_iterations` (int, default=500): Maximum convergence iterations
        - `freeze_constant` (bool, default=False): Freeze constant nodes
        - `verbose` (bool, default=False): Whether to print steady state information
        - `seed` (int, optional): Random seed for reproducibility
        """
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            
        # Store original state
        self._save_network_state()
        
        orig_input_idx = self.input_indices.copy()
        if not freeze_constant:
            # strip out nodes that are "input" only because of A=A
            self.input_indices = [i for i in orig_input_idx if not self._is_self_dependent(i)]
        else:
            # When freeze_constant=True, also include knocked-out nodes as inputs
            knocked_out_nodes = [i for i in range(self.N) if self._is_constant_node(i)]
            self.input_indices = list(set(orig_input_idx + knocked_out_nodes))
        
        # Initialize parameters
        m0 = 0  # burn-in steps
        nsteps = initial_nsteps
        N_memory = 0
        m0_memory = 0
        
        # Track all measured nodes except input for convergence
        opt_states = [i for i in range(self.N) if i not in self.input_indices]
        
        if not opt_states:
            # All nodes are inputs/constants
            steady_state = self.network.nodes.astype(float)
            # Restore original state and undo knockouts
            self._restore_network_state()
            self.network.undoKnockouts()
            self.input_indices = orig_input_idx
            return steady_state

        N_collect = np.zeros(len(opt_states))
        m0_collect = np.zeros(len(opt_states))
        
        counting = 1
        y_kept = []
        Collection = []
        
        while len(y_kept) < nsteps and counting <= max_iterations:
            if counting == 1:
                # First simulation round
                Collection.append([N_collect.copy(), m0_collect.copy(), N_memory, m0, nsteps])
                
                # Initial condition
                initial_state = self.network.nodes.copy()
                for i in range(self.N):
                    if i not in self.input_indices:
                        initial_state[i] = int(np.random.rand() > 0.5)
            
                # Run simulation
                y_trajectory = self._run_tsmc_simulation(initial_state, nsteps, p_mir, p_noise)
                
            else:
                # Continue from last state
                if len(y_kept) > 0:
                    initial_state = np.array(y_kept[-1], dtype=np.int8)
                else:
                    initial_state = (np.random.rand(self.N) > 0.5).astype(np.int8)
                
                # Run additional steps
                remaining_steps = nsteps - len(y_kept) + 1
                y_trajectory = self._run_tsmc_simulation(initial_state, remaining_steps, p_mir, p_noise)
            
            # Append new trajectory (excluding initial state)
            if counting == 1:
                y_kept = y_trajectory[1:].tolist()
            else:
                y_kept.extend(y_trajectory[1:].tolist())
            
            # Convert to numpy array for analysis
            y_array = np.array(y_kept)
            
            # Calculate required nsteps and m0 using marginal distribution analysis
            nsteps, m0, Collection = self._two_state_mc_marg_dist(
                y_array, nsteps, m0, epsilon, r, s, opt_states, 
                Collection, N_memory, m0_memory, N_collect, m0_collect
            )
            
            counting += 1
            
        if counting > max_iterations:
            warnings.warn(f"TSMC did not converge after {max_iterations} iterations")
        
        # Calculate final steady-state probabilities
        if len(y_kept) > 0:
            y_final = np.array(y_kept)
            steady_state = np.zeros(self.N)
            
            # Save knockout values before processing
            knockout_values = {}
            for i in range(self.N):
                if self._is_constant_node(i):
                    knockout_values[i] = self.network.nodes[i]
            
            # Handle different types of nodes
            for i in range(self.N):
                if i in self.input_indices:
                    # For input nodes, use their constant value
                    steady_state[i] = self.network.nodes[i]
                elif self._is_constant_node(i):
                    # For knocked-out nodes, use their fixed knockout value
                    steady_state[i] = knockout_values[i]
                else:
                    # For regular nodes, use trajectory average
                    steady_state[i] = np.mean(y_final[:, i])
        else:
            # Fallback to Monte Carlo if TSMC fails
            warnings.warn("TSMC failed, falling back to Monte Carlo")
            steady_state = self.compute_stationary_mc(seed=seed)
        
        # Restore original state and undo knockouts
        self._restore_network_state()
        self.network.undoKnockouts()
        self.input_indices = orig_input_idx
        if verbose:
            print("Steady state:")
            for i, node in enumerate(self.network.nodeDict.keys()):
                print(f"{node}: {steady_state[i]:.4f}")

        return steady_state
    
    def _run_tsmc_simulation(self, initial_state: np.ndarray, nsteps: int, p_mir: float, p_noise: float) -> np.ndarray:
        """
        Run TSMC simulation
        """
        y = np.zeros((nsteps + 1, self.N), dtype=np.int8)
        y[0] = initial_state
        
        # Set initial state in network
        self.network.setInitialValues(initial_state)
        
        for step in range(nsteps):
            if p_noise > 0:
                trajectory = self.network.update_noise(p_noise, 1)  # Single step
            else:
                trajectory = self.network.update(1)  # Single step
            y_next = trajectory[-1].copy()  # Get the updated state
            
            # Apply Miranda perturbation (but not to input nodes)
            if p_mir > 0:
                pert = np.random.rand(self.N) <= p_mir
                pert[self.input_indices] = False  # Don't perturb inputs
                y_next = np.mod(y_next + pert, 2).astype(np.int8)
                
                # Update network state with perturbation
                self.network.setInitialValues(y_next)
            
            y[step + 1] = y_next
            
        return y
    
    def _two_state_mc_marg_dist(self, y: np.ndarray, nsteps: int, m0: int, 
                               epsilon: float, r: float, s: float, opt_states: List[int],
                               Collection: List, N_memory: int, m0_memory: int,
                               N_collect: np.ndarray, m0_collect: np.ndarray) -> Tuple[int, int, List]:
        """
        Two-State Markov Chain marginal distribution analysis
        In addition to the original function, handle constant/deterministic nodes
        
        Ref: TwoStateMC_MargDist.m
        """
        from scipy.stats import norm
        
        for counter, state_idx in enumerate(opt_states):
            if len(y) <= m0 + 1:
                continue
                
            y_current = y[:, state_idx]
            
            # Calculate transition counts
            y_crosstab = np.column_stack([y_current[m0:-1], y_current[m0+1:]])
            
            if len(y_crosstab) == 0:
                continue
                
            pre1 = np.sum(y_crosstab[:, 0])
            counts = np.zeros((2, 2))
            
            counts[1, 1] = np.sum(y_crosstab[:, 0] * y_crosstab[:, 1])
            counts[1, 0] = pre1 - counts[1, 1]
            counts[0, 0] = np.sum((y_crosstab[:, 0] - 1) * (y_crosstab[:, 1] - 1))
            counts[0, 1] = len(y_crosstab) - counts[0, 0] - counts[1, 0] - counts[1, 1]
            
            # Calculate transition probabilities
            row_sums = np.sum(counts, axis=1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            pp = counts / row_sums[:, np.newaxis]
            
            # Extract alpha and beta
            alpha = pp[0, 1] if row_sums[0] > 0 else 0
            beta = pp[1, 0] if row_sums[1] > 0 else 0
            
            # Calculate m0 and N
            if alpha + beta > 0 and abs(1 - alpha - beta) > 1e-10:
                m0_temp = np.log10(epsilon * (alpha + beta) / max(alpha, beta)) / np.log10(abs(1 - alpha - beta))
                N_temp = (alpha * beta * (2 - alpha - beta) / 
                         (alpha + beta)**3 * (r / norm.ppf(0.5 * (1 + s)))**(-2))
                
                m0_collect[counter] = max(0, np.real(m0_temp))
                N_collect[counter] = max(0, N_temp)
            
        # Update memory values
        N_memory = max(np.max(N_collect), N_memory)
        m0_memory = max(np.max(m0_collect), m0_memory)
        
        # Update nsteps and m0 if needed
        if nsteps < N_memory:
            nsteps = int(np.ceil(N_memory))
            m0 = int(np.ceil(m0_memory))
            
        # Collect results
        Collection.append([N_collect.copy(), m0_collect.copy(), N_memory, m0, nsteps])
        
        return nsteps, m0, Collection
    
    def compute_stationary_mc(self, 
                             n_runs: int = 10, 
                             n_steps: int = 1000,
                             p_noise: float = 0,
                             analyze_convergence: bool = False,
                             output_node: str = None,
                             verbose: bool = False,
                             seed: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Monte Carlo steady-state calculation, handle input nodes and constant nodes
        
        Parameters:
        -----------
        n_runs : int, default=10
            Number of independent Monte Carlo runs
        n_steps : int, default=1000
            Number of simulation steps per run
        p_noise : float, default=0
            Noise probability for simulation
        analyze_convergence : bool, default=False
            Whether to perform convergence analysis and show plot
        output_node : str, optional
            Node name for convergence analysis. If None, uses all nodes.
        verbose : bool, default=False
            Whether to print steady state information
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray or Tuple[np.ndarray, Dict]
            If analyze_convergence is False: steady-state probabilities
            If analyze_convergence is True: (steady_state, convergence_info)
        """
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            
        # Store original state
        self._save_network_state()
        
        if analyze_convergence:
            # Single long run for convergence analysis
            steady_state, convergence_info = self._compute_mc_with_convergence(n_steps, p_noise, output_node, seed)
            
            # convergence plot
            self._convergence_plot(convergence_info, output_node)
            
            # Restore original state and undo knockouts
            self._restore_network_state()
            self.network.undoKnockouts()
            
            return steady_state, convergence_info
        else:
            # Standard multiple runs approach
            steady_states = []
            
            # Save knockout values before simulation
            knockout_values = {}
            for i in range(self.N):
                if self._is_constant_node(i):
                    knockout_values[i] = self.network.nodes[i]
            
            for run in range(n_runs):
                # Random initial state
                initial_state = self.network.nodes.copy()
                for i in range(self.N):
                    if i not in self.input_indices:
                        initial_state[i] = int(np.random.rand() > 0.5)
                self.network.setInitialValues(initial_state)
                
                # Run simulation
                trajectory = self.network.update_noise(p=p_noise, iterations=n_steps)
                
                # Take second half as steady state
                steady_portion = trajectory[n_steps//2:]
                mean_state = np.mean(steady_portion, axis=0)
                
                # Fix constant/knocked-out nodes to their original knockout values
                for i in range(self.N):
                    if self._is_constant_node(i):
                        mean_state[i] = knockout_values[i]
                
                steady_states.append(mean_state)
            
            # Calculate final steady state
            final_steady_state = np.mean(steady_states, axis=0)
            
            # Ensure constant nodes have their fixed knockout values
            for i in range(self.N):
                if self._is_constant_node(i):
                    final_steady_state[i] = knockout_values[i]
            
            # Restore original state and undo knockouts
            self._restore_network_state()
            self.network.undoKnockouts()
            
            if verbose:
                print("Steady state:")
                for i, node in enumerate(self.network.nodeDict.keys()):
                    print(f"{node}: {final_steady_state[i]:.4f}")

            return final_steady_state
    
    def _compute_mc_with_convergence(self, n_steps: int, p_noise: float, output_node: str = None, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Compute Monte Carlo steady state with convergence analysis.
        
        Parameters:
        -----------
        n_steps : int
            Number of simulation steps
        p_noise : float
            Noise probability
        output_node : str, optional
            Node name for convergence analysis
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple[np.ndarray, Dict]
            (steady_state, convergence_info)
        """
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            
        # Random initial state
        initial_state = self.network.nodes.copy()
        for i in range(self.N):
            if i not in self.input_indices:
                initial_state[i] = int(np.random.rand() > 0.5)
        self.network.setInitialValues(initial_state)
        
        # Run single long trajectory
        trajectory = self.network.update_noise(p=p_noise, iterations=n_steps)
        
        # Take second half for steady state calculation
        steady_portion = trajectory[n_steps//2:]
        steady_state = np.mean(steady_portion, axis=0)
        
        # Save and apply knockout values
        knockout_values = {}
        for i in range(self.N):
            if self._is_constant_node(i):
                knockout_values[i] = self.network.nodes[i]
        
        # Fix constant/knocked-out nodes to their original knockout values
        for i in range(self.N):
            if self._is_constant_node(i):
                steady_state[i] = knockout_values[i]
        
        # Convergence analysis on second half
        convergence_info = self._analyze_convergence(steady_portion, output_node)
        
        return steady_state, convergence_info
    
    def _analyze_convergence(self, trajectory: np.ndarray, output_node: str = None) -> Dict:
        """
        Analyze convergence of Monte Carlo trajectory.
        
        Based on the approach in steady_state_convergence.py:
        - Calculate running averages at different time points
        - Compute relative changes between consecutive averages
        - Report final relative change as convergence measure
        
        Parameters:
        -----------
        trajectory : np.ndarray
            Trajectory data (second half of simulation)
        output_node : str, optional
            Node name for convergence analysis. If None, analyzes all nodes.
            
        Returns:
        --------
        Dict
            Convergence analysis results
        """
        if len(trajectory) < 10:
            return {'final_relative_change': 0.0, 'converged': True}
        
        # Define time points for convergence analysis
        traj_length = len(trajectory)
        step_size = max(1, traj_length // 30)  # About 30 time points
        time_points = list(range(step_size, traj_length + 1, step_size))
        
        if output_node and output_node in self.network.nodeDict:
            # Analyze specific node
            node_idx = self.network.nodeDict[output_node]
            node_scores = self._calculate_node_convergence(trajectory, node_idx, time_points)
            final_relative_change = node_scores[-1] if node_scores else 0.0
            
            convergence_info = {
                'final_relative_change': final_relative_change,
                'converged': final_relative_change < 1.0,  # Less than 1% change
                'node_analyzed': output_node,
                'relative_changes': node_scores,
                'time_points': [i*2 for i in time_points]
            }
        else:
            # Analyze all nodes and take maximum
            all_relative_changes = []
            
            for node_idx in range(self.N):
                if node_idx not in self.input_indices:  # Skip input nodes
                    node_scores = self._calculate_node_convergence(trajectory, node_idx, time_points)
                    if node_scores:
                        all_relative_changes.extend(node_scores)
            
            final_relative_change = all_relative_changes[-1] if all_relative_changes else 0.0
            
            convergence_info = {
                'final_relative_change': final_relative_change,
                'converged': final_relative_change < 1.0,  # Less than 1% change
                'node_analyzed': 'all_nodes',
                'n_nodes_analyzed': self.N - len(self.input_indices)
            }
        
        return convergence_info
    
    def _convergence_plot(self, convergence_info: Dict, output_node: str = None):
        """
        Display convergence plot
        
        Parameters:
        -----------
        convergence_info : Dict
            Convergence information from _analyze_convergence
        output_node : str, optional
            Node name for plot title
        """
        import matplotlib.pyplot as plt
        
        if 'relative_changes' in convergence_info and 'time_points' in convergence_info:
            relative_changes = convergence_info['relative_changes']
            time_points = convergence_info['time_points']
            
            if relative_changes and len(relative_changes) > 0:
                plt.figure(figsize=(8, 5))
                plt.scatter(time_points[1:], relative_changes, alpha=0.7)
                plt.xlabel('Simulation Steps')
                plt.ylabel('Relative Change of Score in %')
                
                node_name = output_node if output_node else 'All Nodes'
                plt.title(f'Convergence Analysis - {node_name}')
                plt.grid(True, alpha=0.3)
                
                # Add horizontal line at 1% for reference
                plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1% threshold')
                
                # Add final convergence value as text
                final_change = convergence_info['final_relative_change']
                plt.text(0.02, 0.98, f'Final: {final_change:.2f}%', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.legend()
                plt.tight_layout()
                plt.show()
    
    def _calculate_node_convergence(self, trajectory: np.ndarray, node_idx: int, time_points: List[int]) -> List[float]:
        """
        Calculate convergence for a specific node.
        
        Parameters:
        -----------
        trajectory : np.ndarray
            Trajectory data
        node_idx : int
            Node index to analyze
        time_points : List[int]
            Time points for convergence analysis
            
        Returns:
        --------
        List[float]
            Relative changes in percentage
        """
        scores = []
        
        # Calculate running averages at different time points
        for n in time_points:
            if n <= len(trajectory):
                # Take average from start to time point n
                avg_value = np.mean(trajectory[:n, node_idx])
                scores.append(avg_value)
        
        # Calculate relative changes between consecutive scores
        relative_changes = []
        for i in range(1, len(scores)):
            diff = scores[i] - scores[i-1]
            relative_change = 100 * abs(diff)  # Convert to percentage
            relative_changes.append(relative_change)
        
        return relative_changes
    
    def set_experimental_conditions(self, stimuli: List[str] = None, 
                                   stimuli_efficacy: List[float] = None,
                                   inhibitors: List[str] = None,
                                   inhibitors_efficacy: List[float] = None,
                                   node_dict: Dict[str, int] = None):
        """
        Set experimental conditions for steady state calculation.
        First sets all input nodes to 0 by default, then applies specified stimuli and inhibitors.
        
        Parameters:
        -----------
        stimuli : List[str], optional
            Node names to stimulate (fix to 1)
        stimuli_efficacy : List[float], optional
            Efficacy values for stimuli (0-1). If 1.0, full knockout. If < 1.0, probabilistic.
        inhibitors : List[str], optional
            Node names to inhibit (fix to 0)
        inhibitors_efficacy : List[float], optional
            Efficacy values for inhibitors (0-1). If 1.0, full knockout. If < 1.0, probabilistic.
        node_dict : Dict[str, int], optional
            Node name to index mapping
            
        Note:
        -----
        This method automatically sets all input nodes to 0 by default before applying 
        the specified experimental conditions. Input nodes not specified in stimuli 
        or inhibitors will remain at 0.
        """
        if node_dict is None:
            node_dict = getattr(self.network, 'nodeDict', {})
        
        # First, set all input nodes to 0 by default
        self._set_input_nodes_to_zero()
        
        # Apply stimuli with efficacy
        if stimuli:
            stimuli_eff = stimuli_efficacy if stimuli_efficacy else [1.0] * len(stimuli)
            for node_name, efficacy in zip(stimuli, stimuli_eff):
                if node_name in node_dict:
                    if efficacy == 1.0:
                        self.network.knockout(node_name, 1)
                    else:
                        self.network.knockdown(node_name, 1, efficacy)
        
        # Apply inhibitors with efficacy
        if inhibitors:
            inhibitors_eff = inhibitors_efficacy if inhibitors_efficacy else [1.0] * len(inhibitors)
            for node_name, efficacy in zip(inhibitors, inhibitors_eff):
                if node_name in node_dict:
                    if efficacy == 1.0:
                        self.network.knockout(node_name, 0)
                    else:
                        self.network.knockdown(node_name, 0, efficacy)
    
    def reset_network_conditions(self):
        """Reset network to original state"""
        # Reset both knockouts and knockdowns
        self.network.undoKnockouts()
    
    def _save_network_state(self):
        """Save the current network state for restoration"""
        self._original_nodes = self.network.nodes.copy()
    
    def _restore_network_state(self):
        """Restore the saved network state"""
        if self._original_nodes is not None:
            self.network.setInitialValues(self._original_nodes)
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """
        Get information about convergence properties
        """
        return {
            'network_type': 'PBN' if self.is_pbn else 'BN',
            'num_nodes': self.N,
            'input_nodes': self.input_indices,
            'num_functions': self.Nf if self.is_pbn else self.N,
            'max_connectivity': np.max(self.K) if len(self.K) > 0 else 0
        }

    def compute_stationary_deterministic(self, n_runs: int = 100, n_steps: int = 1000, verbose: bool = True, seed: Optional[int] = None) -> Dict[str, Union[List[np.ndarray], List[List[np.ndarray]]]]:
        """
        Find attractors (fixed points and cycles) in a synchronous
        Boolean network via random restarts.
        
        Parameters:
        -----------
        n_runs : int, default=100
            Number of random initial conditions to try
        n_steps : int, default=1000
            Maximum number of steps to simulate before declaring no cycle found
        verbose : bool, default=True
            Whether to print steady state information
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        {
            "fixed_points": [state_vec, ...],                  # each np.ndarray (0/1)
            "cyclic_attractors"      : [[state_vec1, state_vec2, ...],    # one list per cycle
                            ...]
        }
        """
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            
        fixed_points: List[np.ndarray] = []
        cycles: List[List[np.ndarray]] = []

        def _canonical_cycle(cycle: List[Tuple[int, ...]]) -> Tuple[Tuple[int, ...], ...]:
            """Return a rotation-invariant representation of a cycle."""
            # choose lexicographically smallest rotation
            rotations = [cycle[i:] + cycle[:i] for i in range(len(cycle))]
            return tuple(min(rotations))

        seen_fixed: set[Tuple[int, ...]] = set()
        seen_cycles: set[Tuple[Tuple[int, ...], ...]] = set()

        self._save_network_state()  # keep original state

        for _ in range(n_runs):
            # random start
            init = (np.random.rand(self.N) > 0.5).astype(np.int8)
            # Keep input nodes at their current values (set by experimental conditions)
            for i in self.input_indices:
                init[i] = self.network.nodes[i]
            
            self.network.setInitialValues(init)

            history: List[Tuple[int, ...]] = []

            for _ in range(n_steps):
                state = tuple(self.network.nodes.copy())

                if state in history:   
                    start_idx = history.index(state)
                    cycle = history[start_idx:]  # list[tuple]

                    if len(cycle) == 1:          # fixed point
                        if state not in seen_fixed:
                            fixed_points.append(np.array(state))
                            seen_fixed.add(state)
                    else:                        # limit cycle
                        can = _canonical_cycle(cycle)
                        if can not in seen_cycles:
                            cycles.append([np.array(s) for s in cycle])
                            seen_cycles.add(can)
                    break  # stop this trajectory

                history.append(state)
                self.network.update(1)

        # Restore original state and undo knockouts
        self._restore_network_state()
        self.network.undoKnockouts()
        
        if verbose:
            print(f"Found {len(fixed_points)} fixed points and {len(cycles)} cyclic attractors")
            print("--------------------------------")
            if len(fixed_points) > 0:
                print("Fixed points: ")
                for i, fixed_point in enumerate(fixed_points):
                    print(f"Fixed point {i+1}: {fixed_point.tolist()}")
            else:
                print("No fixed points found")
            print("--------------------------------")
            if len(cycles) > 0:
                print("Cyclic attractors: ")
                for i, cycle in enumerate(cycles):
                    print(f"Cyclic attractor {i+1}: {[component.tolist() for component in cycle]}")
            else:
                print("No cyclic attractors found")
            print("--------------------------------")
            print(f"Node order: {self.network.nodeDict.keys()}")
        return {"fixed_points": fixed_points, "cyclic_attractors": cycles}

    def _is_constant_node(self, node_idx: int) -> bool:
        """Check if a node is a constant/knocked-out node"""
        if hasattr(self.network, 'isConstantNode'):
            return self.network.isConstantNode[node_idx]
        return False