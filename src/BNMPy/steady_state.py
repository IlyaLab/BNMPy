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
        - sum(nf*nv) == 0 (no inputs)
        - sum(nf*nv) == 1 and self-dependent (self-dependent only)

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
            
            # Check if input: sum=0 (no inputs) or sum=1 and self-dependent
            if nfnv_sum == 0 or (nfnv_sum == 1 and self._is_self_dependent(i)):
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
    
    def compute_steady_state(self, method: str = 'tsmc', **kwargs) -> np.ndarray:
        """
        steady-state calculation
        
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
                               initial_nsteps: int = 1000,
                               max_iterations: int = 100,
                               freeze_self_loop: bool = False) -> np.ndarray:
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
        - `initial_nsteps` (int, default=1000): Initial number of simulation steps
        - `max_iterations` (int, default=100): Maximum convergence iterations
        - `freeze_self_loop` (bool, default=False): Freeze self-loop nodes (constant nodes)
        """
        # Store original state
        self._save_network_state()
        orig_input_idx = self.input_indices.copy()
        if not freeze_self_loop:
            # strip out nodes that are "input" only because of A=A
            self.input_indices = [i for i in orig_input_idx if not self._is_self_dependent(i)]
        
        # Initialize parameters
        m0 = 0  # burn-in steps
        nsteps = initial_nsteps
        N_memory = 0
        m0_memory = 0
        
        # Track all measured nodes except input for convergence
        opt_states = [i for i in range(self.N) if i not in self.input_indices]
        
        if not opt_states:
            # All nodes are inputs/constants
            return self.network.nodes.astype(float)

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
            # For input nodes, use their constant value
            for i in self.input_indices:
                steady_state[i] = self.network.nodes[i]  # or initial value
            
            # For other nodes, use the average of all trajectories
            for i in range(self.N):
                if i not in self.input_indices:
                    steady_state[i] = np.mean(y_final[:, i])
        else:
            # Fallback to Monte Carlo if TSMC fails
            warnings.warn("TSMC failed, falling back to Monte Carlo")
            steady_state = self.compute_stationary_mc()
        
        # Restore original state
        self._restore_network_state()
        self.input_indices = orig_input_idx
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
                             p_noise: float = 0) -> np.ndarray:
        """
        Monte Carlo steady-state calculation, handle input nodes and constant nodes
        """
        # Store original state
        self._save_network_state()
        
        steady_states = []
        
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
            steady_states.append(mean_state)
        
        # Restore original state
        self._restore_network_state()
        
        return np.mean(steady_states, axis=0)
    
    def set_experimental_conditions(self, stimuli: List[str] = None, 
                                   inhibitors: List[str] = None,
                                   node_dict: Dict[str, int] = None):
        """
        Set experimental conditions
        """
        if node_dict is None:
            node_dict = getattr(self.network, 'nodeDict', {})
        
        # Apply stimuli (knockouts to 1)
        if stimuli:
            for node_name in stimuli:
                if node_name in node_dict:
                    self.network.knockout(node_name, 1)
        
        # Apply inhibitors (knockouts to 0)
        if inhibitors:
            for node_name in inhibitors:
                if node_name in node_dict:
                    self.network.knockout(node_name, 0)
    
    def reset_network_conditions(self):
        """Reset network to original state"""
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

    def compute_stationary_deterministic(self, max_steps: int = 1000) -> Union[np.ndarray, None]:
        """
        Find deterministic steady state (attractors) for Boolean networks
        TODO: need more discussion on this
        """
        # Store original state
        self._save_network_state()
        
        # Try different initial conditions
        for _ in range(10):  # Try up to 10 random initial conditions
            initial_state = (np.random.rand(self.N) > 0.5).astype(np.int8)
            self.network.setInitialValues(initial_state)
            
            # Track states to detect cycles
            state_history = []
            
            for step in range(max_steps):
                current_state = self.network.nodes.copy()
                state_tuple = tuple(current_state)
                
                if state_tuple in state_history:
                    # Found a cycle
                    cycle_start = state_history.index(state_tuple)
                    cycle = state_history[cycle_start:]
                    
                    if len(cycle) == 1:
                        # Fixed point found
                        self._restore_network_state()
                        return current_state.astype(float)
                    else:
                        # Limit cycle - return average
                        cycle_states = np.array([list(s) for s in cycle])
                        self._restore_network_state()
                        return np.mean(cycle_states, axis=0)
                
                state_history.append(state_tuple)
                
                # Update network using existing method
                self.network.update(1)
        
        # Restore original state and return None if no attractor found
        self._restore_network_state()
        return None