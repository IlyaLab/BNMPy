import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from SALib.sample import saltelli, morris as morris_sampler
from SALib.analyze import sobol, morris
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
from dataclasses import dataclass


from KGBN.steady_state import SteadyStateCalculator

def sensitivity_analysis(network, experiments, config=None, top_n=5, verbose=True):
    """
    Perform sensitivity analysis to identify the most influential nodes affecting measurements.
    
    Parameters:
    -----------
    network : ProbabilisticBN
        The PBN object to analyze
    experiments : str or list
        Path to experiments CSV or list of experiment dictionaries
    config : dict, optional
        Configuration including steady state parameters and sensitivity method
    top_n : int
        Number of top sensitive nodes to return
    verbose : bool
        Whether to print detailed progress information
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'top_nodes': List of top N sensitive node names
        - 'sensitivity_df': DataFrame with all sensitivity results
    """
    start_time = time.time()
    
    # Initialize analyzer
    analyzer = PBNSensitivityAnalyzer(network, experiments, config, verbose)
    
    if verbose:
        print("="*60)
        print("PBN SENSITIVITY ANALYSIS")
        print("="*60)
    
    # Run sensitivity analysis
    sensitivity_df = analyzer.run_analysis()
    
    # Get top nodes
    top_nodes = analyzer.get_top_sensitive_nodes(sensitivity_df, top_n)
    
    analysis_time = time.time() - start_time
    
    if verbose:
        print(f"\nTotal analysis time: {analysis_time:.2f} seconds")
        print("="*60)
    
    return {
        'top_nodes': top_nodes,
        'sensitivity_df': sensitivity_df
    }


class PBNSensitivityAnalyzer:
    """
    Performs sensitivity analysis on PBN using existing steady state methods.
    """
    
    def __init__(self, network, experiments, config=None, verbose=True):
        self.pbn = network
        self.verbose = verbose
        
        # Load experiments if path provided
        if isinstance(experiments, str):
            from KGBN.experiment_data import ExperimentData
            self.experiments = ExperimentData.load_from_csv(experiments)
        else:
            self.experiments = experiments
            
        # Set up configuration
        self.config = self._default_config()
        if config:
            self.config.update(config)
            
        # Initialize simulation evaluator for steady state calculations
        from KGBN.simulation_evaluator import SimulationEvaluator
        self.evaluator = SimulationEvaluator(
            self.pbn, 
            self.experiments, 
            self.config
        )
        
        # Extract measured nodes from experiments
        self.measured_nodes = self._get_measured_nodes()
        
        # Determine which nodes can be analyzed (not measured, have multiple functions)
        self.analyzable_nodes = self._get_analyzable_nodes()
        
        # Store original Cij for faster restoration
        self.original_cij = self.pbn.cij.copy()
    
    def _default_config(self):
        """Default configuration"""
        return {
            'sensitivity_method': 'morris',  # 'morris', or 'sobol'
            'morris_trajectories': 10,    
            'sobol_samples': 512,         
            'sobol_second_order': False,
            'parallel': True,             
            'n_workers': -1,                      # use all CPUs
            'batch_size': 50,                     # Batch size for parallel processing
            'seed': 9,                            # Global seed for reproducibility
            'steady_state': {
                'method': 'tsmc',
                'tsmc_params': {
                    'epsilon': 0.01,        
                    'r': 0.05,               
                    's': 0.90,              
                    'p_mir': 0.001,
                    'initial_nsteps': 100,       
                    'max_iterations': 1000,       
                },
                'monte_carlo_params': {
                    'n_runs': 2,              
                    'n_steps': 5000,           
                    'p_noise': 0.05
                }
            }
        }
    
    def _get_measured_nodes(self):
        """Extract all measured nodes from experiments"""
        measured = set()
        for exp in self.experiments:
            measured.update(exp['measurements'].keys())
        return list(measured)
    
    def _get_analyzable_nodes(self):
        """Get nodes that can be analyzed (not measured, have variable functions)"""
        analyzable = []
        for node_name, node_idx in self.pbn.nodeDict.items():
            # Skip measured nodes
            if node_name in self.measured_nodes:
                continue
            # Skip nodes with only one function (no variability)
            if self.pbn.nf[node_idx] <= 1:
                continue
            analyzable.append(node_name)
        return analyzable
    
    def run_analysis(self):
        """Run the sensitivity analysis"""
        if self.verbose:
            print(f"\nNetwork information:")
            print(f"  - Total nodes: {len(self.pbn.nodeDict)}")
            print(f"  - Measured nodes: {len(self.measured_nodes)} ({', '.join(self.measured_nodes)})")
            print(f"  - Analyzable nodes: {len(self.analyzable_nodes)}")
            print(f"  - Experiments: {len(self.experiments)}")
        
        method = self.config.get('sensitivity_method', 'fast_morris')
        
        if self.verbose:
            print(f"\nUsing {method.upper()} sensitivity analysis method")
            if self.config.get('parallel', True):
                n_workers = self.config.get('n_workers', -1)
                if n_workers == -1:
                    n_workers = cpu_count()
                print(f"  - Parallel processing: Enabled ({n_workers} workers)")
            
        if method == 'fast_morris':
            return self._run_fast_morris_analysis()
        elif method == 'morris':
            return self._run_morris_analysis()
        elif method == 'sobol':
            return self._run_sobol_analysis()
        else:
            raise ValueError(f"Unknown sensitivity method: {method}")
    
    def _create_problem(self):
        """Create SALib problem definition"""
        problem = {
            'num_vars': len(self.analyzable_nodes),
            'names': [],
            'bounds': [],
            'node_info': []  # Store (node_name, node_idx) for each parameter
        }
        
        for node_name in self.analyzable_nodes:
            node_idx = self.pbn.nodeDict[node_name]
            # Each parameter represents the selection probability adjustment
            problem['names'].append(node_name)
            problem['bounds'].append([0, 1])
            problem['node_info'].append((node_name, node_idx))
            
        return problem
    
    def _run_morris_analysis(self):
        """Run Morris analysis"""
        problem = self._create_problem()
        trajectories = self.config.get('morris_trajectories', 20)
        
        if self.verbose:
            print(f"\nMorris analysis setup:")
            print(f"  - Parameters: {problem['num_vars']}")
            print(f"  - Trajectories: {trajectories}")
            print(f"  - Total evaluations: ~{trajectories * (problem['num_vars'] + 1)}")
        
        # Generate samples
        X = morris_sampler.sample(problem, trajectories, num_levels=4)
        
        # Evaluate model
        if self.config.get('parallel', True):
            Y = self._evaluate_samples_parallel(X, problem)
        else:
            Y = self._evaluate_samples(X, problem)
        
        # Analyze for each measured node
        results = []
        for j, measured_node in enumerate(self.measured_nodes):
            Si = morris.analyze(problem, X, Y[:, j], print_to_console=False)
            
            for i, node_name in enumerate(problem['names']):
                results.append({
                    'node': node_name,
                    'measured_node': measured_node,
                    'mu': Si['mu'][i],
                    'mu_star': Si['mu_star'][i],
                    'sigma': Si['sigma'][i]
                })
        
        df = pd.DataFrame(results)
        return df
    
    def _run_sobol_analysis(self):
        """Run Sobol variance-based analysis"""
        problem = self._create_problem()
        n_samples = self.config.get('sobol_samples', 512)
        calc_second = self.config.get('sobol_second_order', False)
        
        if self.verbose:
            print(f"\nSobol analysis setup:")
            print(f"  - Parameters: {problem['num_vars']}")
            print(f"  - Base samples: {n_samples}")
            multiplier = 2 * problem['num_vars'] + 2 if calc_second else problem['num_vars'] + 2
            print(f"  - Total evaluations: {n_samples * multiplier}")
        
        # Generate samples
        X = saltelli.sample(problem, n_samples, calc_second_order=calc_second)
        
        # Evaluate model
        if self.config.get('parallel', True):
            Y = self._evaluate_samples_parallel(X, problem)
        else:
            Y = self._evaluate_samples(X, problem)
        
        # Analyze for each measured node
        results = []
        for j, measured_node in enumerate(self.measured_nodes):
            Si = sobol.analyze(problem, Y[:, j], calc_second_order=calc_second, print_to_console=False)
            
            for i, node_name in enumerate(problem['names']):
                result_dict = {
                    'node': node_name,
                    'measured_node': measured_node,
                    'S1': Si['S1'][i],
                    'ST': Si['ST'][i]
                }
                if calc_second:
                    result_dict['S2'] = Si['ST'][i] - Si['S1'][i]
                results.append(result_dict)
        
        df = pd.DataFrame(results)
        return df
    
    def _evaluate_single_sample(self, x, problem):
        """Evaluate a single parameter sample"""
        # Convert to Cij
        cij_matrix = self._parameters_to_cij(x, problem)
        
        # Calculate response
        n_measured = len(self.measured_nodes)
        exp_results = np.zeros((len(self.experiments), n_measured))
            
        for exp_idx, experiment in enumerate(self.experiments):
            try:
                # Update PBN parameters
                self.evaluator._update_pbn_parameters(cij_matrix)
                
                # Simulate experiment
                steady_state = self.evaluator._simulate_experiment(experiment)
                
                # Extract measured node values
                for j, node_name in enumerate(self.measured_nodes):
                    node_idx = self.pbn.nodeDict[node_name]
                    exp_results[exp_idx, j] = steady_state[node_idx]
                    
            except Exception as e:
                exp_results[exp_idx, :] = np.nan
        
        # Average across experiments
        return np.nanmean(exp_results, axis=0)
    
    def _evaluate_samples(self, X, problem):
        """Evaluate the model for all parameter samples (sequential)"""
        n_samples = X.shape[0]
        n_measured = len(self.measured_nodes)
        Y = np.zeros((n_samples, n_measured))
        
        if self.verbose:
            print(f"\nEvaluating {n_samples} parameter sets...")
            iterator = tqdm(range(n_samples))
        else:
            iterator = range(n_samples)
        
        for i in iterator:
            Y[i, :] = self._evaluate_single_sample(X[i], problem)
        
        # Restore original parameters
        self.evaluator._update_pbn_parameters(self.original_cij)
        
        return Y
    
    def _evaluate_samples_parallel(self, X, problem):
        """Evaluate the model for all parameter samples (parallel)"""
        n_samples = X.shape[0]
        n_measured = len(self.measured_nodes)
        
        # Determine number of workers
        n_workers = self.config.get('n_workers', -1)
        if n_workers == -1:
            n_workers = min(cpu_count(), n_samples)
        
        if self.verbose:
            print(f"\nEvaluating {n_samples} parameter sets in parallel ({n_workers} workers)...")
        
        # Create worker pool
        # Note: We need to create separate evaluators for each worker
        
        # Batch the samples
        batch_size = max(1, n_samples // (n_workers * 4))  # 4 batches per worker
        batches = []
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size]
            batches.append((batch_X, problem, self.pbn, self.experiments, self.config, self.measured_nodes))
        
        # Process batches in parallel
        with Pool(n_workers) as pool:
            if self.verbose:
                results = list(tqdm(
                    pool.imap(_evaluate_batch_worker, batches),
                    total=len(batches)
                ))
            else:
                results = pool.map(_evaluate_batch_worker, batches)
        
        # Combine results
        Y = np.vstack(results)
        
        # Restore original parameters
        self.evaluator._update_pbn_parameters(self.original_cij)
        
        return Y
    
    def _parameters_to_cij(self, params, problem):
        """
        Convert parameter vector to modified Cij matrix.
        For 2-function nodes: parameter directly controls function selection probability.
        For multi-function nodes: parameter scales the dominant function's probability.
        """
        cij_matrix = self.original_cij.copy()
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            for i, param_value in enumerate(params):
                node_name, node_idx = problem['node_info'][i]
                n_functions = self.pbn.nf[node_idx]
                
                if n_functions == 2:
                    # Direct interpretation: param = P(function 1)
                    cij_matrix[node_idx, 0] = param_value
                    cij_matrix[node_idx, 1] = 1 - param_value
                    
                else:  # n_functions > 2
                    # Get original probabilities
                    orig_probs = self.original_cij[node_idx, :n_functions].copy()
                    
                    # Linear interpolation between original and uniform
                    uniform_probs = np.ones(n_functions) / n_functions
                    
                    # param = 0: use original distribution
                    # param = 1: use uniform distribution
                    new_probs = (1 - param_value) * orig_probs + param_value * uniform_probs
                    
                    # Ensure valid probabilities
                    new_probs = np.maximum(new_probs, 1e-10)
                    new_probs = new_probs / np.sum(new_probs)
                    
                    cij_matrix[node_idx, :n_functions] = new_probs
                
                # Preserve -1 values for unused slots
                for j in range(n_functions, len(cij_matrix[node_idx])):
                    cij_matrix[node_idx, j] = -1
        
        return cij_matrix

    def get_top_sensitive_nodes(self, sensitivity_df, top_n=5):
        """Extract top N sensitive nodes from results"""
        if self.verbose:
            print(f"\nIdentifying top {top_n} sensitive nodes...")
        
        # Determine which metric to use
        if 'ST' in sensitivity_df.columns:  # Sobol
            metric = 'ST'  # Total effect
            metric_name = "Total Effect"
        else:  # Morris
            metric = 'mu_star'  # Mean absolute effect
            metric_name = "Mean |Effect|"
        
        # Aggregate sensitivity across all measured nodes
        node_sensitivity = sensitivity_df.groupby('node')[metric].agg(['mean', 'max', 'std'])
        
        # Sort by mean sensitivity
        node_sensitivity = node_sensitivity.sort_values('mean', ascending=False)
        
        # Get top N
        top_nodes = node_sensitivity.head(top_n).index.tolist()
        
        if self.verbose:
            print(f"\nTop {top_n} sensitive nodes (by average {metric_name}):")
            print("-" * 50)
            for i, node in enumerate(top_nodes):
                stats = node_sensitivity.loc[node]
                print(f"{i+1}. {node:15s} - Avg: {stats['mean']:.4f}, Max: {stats['max']:.4f}, Std: {stats['std']:.4f}")
            
            # Also show per-measured-node breakdown for top nodes
            print(f"\nSensitivity breakdown by measured node:")
            print("-" * 50)
            for node in top_nodes[:min(3, len(top_nodes))]:  # Show details for top 3
                node_data = sensitivity_df[sensitivity_df['node'] == node]
                print(f"\n{node}:")
                for _, row in node_data.iterrows():
                    print(f"  -> {row['measured_node']:10s}: {row[metric]:.4f}")
        
        return top_nodes


# Worker function for parallel processing (must be at module level for pickling)
def _evaluate_batch_worker(args):
    """Worker function for parallel batch evaluation"""
    batch_X, problem, pbn, experiments, config, measured_nodes = args
    
    # Suppress warnings in worker processes
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Create local evaluator
    from KGBN.simulation_evaluator import SimulationEvaluator
    evaluator = SimulationEvaluator(pbn, experiments, config)
    
    n_samples = batch_X.shape[0]
    n_measured = len(measured_nodes)
    Y = np.zeros((n_samples, n_measured))
    
    for i in range(n_samples):
        # Convert parameters to Cij
        cij_matrix = _parameters_to_cij_worker(batch_X[i], problem, pbn)
        
        # Calculate average response across experiments
        exp_results = np.zeros((len(experiments), n_measured))
        
        for exp_idx, experiment in enumerate(experiments):
            try:
                evaluator._update_pbn_parameters(cij_matrix)
                steady_state = evaluator._simulate_experiment(experiment)
                
                for j, node_name in enumerate(measured_nodes):
                    node_idx = pbn.nodeDict[node_name]
                    exp_results[exp_idx, j] = steady_state[node_idx]
                    
            except Exception:
                exp_results[exp_idx, :] = np.nan
        
        Y[i, :] = np.nanmean(exp_results, axis=0)
    
    return Y

def _parameters_to_cij_worker(params, problem, pbn):
    """Parameter conversion for worker processes"""
    cij_matrix = pbn.cij.copy()
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        for i, param_value in enumerate(params):
            node_name, node_idx = problem['node_info'][i]
            n_functions = pbn.nf[node_idx]
            
            if n_functions == 2:
                # Direct mapping for 2-function nodes
                cij_matrix[node_idx, 0] = param_value
                cij_matrix[node_idx, 1] = 1 - param_value
                
            else:  # n_functions > 2
                # Linear interpolation approach
                orig_probs = pbn.cij[node_idx, :n_functions].copy()
                uniform_probs = np.ones(n_functions) / n_functions
                
                new_probs = (1 - param_value) * orig_probs + param_value * uniform_probs
                new_probs = np.maximum(new_probs, 1e-10)
                new_probs = new_probs / np.sum(new_probs)
                
                cij_matrix[node_idx, :n_functions] = new_probs
            
            # Preserve -1 values
            for j in range(n_functions, len(cij_matrix[node_idx])):
                cij_matrix[node_idx, j] = -1
    
    return cij_matrix

def plot_sensitivity_results(sensitivity_df, top_n=20, save_path=None):
    """
    Create visualization of sensitivity analysis results.
    
    Parameters:
    -----------
    sensitivity_df : pd.DataFrame
        Results from sensitivity analysis
    top_n : int
        Number of top nodes to display
    save_path : str, optional
        Path to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Determine metric
    if 'ST' in sensitivity_df.columns:
        metric = 'ST'
        metric_name = 'Total Effect Index'
    else:
        metric = 'mu_star'
        metric_name = 'Morris μ*'
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Average sensitivity by node
    node_avg = sensitivity_df.groupby('node')[metric].mean().sort_values(ascending=False).head(top_n)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(node_avg)))
    bars = ax1.barh(range(len(node_avg)), node_avg.values, color=colors)
    ax1.set_yticks(range(len(node_avg)))
    ax1.set_yticklabels(node_avg.index)
    ax1.set_xlabel(f'Average {metric_name}')
    ax1.set_title(f'Top {top_n} Nodes by Average Sensitivity')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, node_avg.values)):
        ax1.text(value + 0.001, i, f'{value:.3f}', va='center', fontsize=8)
    
    # Plot 2: Heatmap for top nodes
    top_nodes = node_avg.head(10).index
    heatmap_data = sensitivity_df[sensitivity_df['node'].isin(top_nodes)].pivot(
        index='node', columns='measured_node', values=metric
    )
    
    # Reorder to match bar chart
    heatmap_data = heatmap_data.loc[top_nodes]
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': metric_name}, ax=ax2)
    ax2.set_title('Sensitivity by Measured Node')
    ax2.set_xlabel('Measured Node')
    ax2.set_ylabel('Parameter Node')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


# Removed simple_sensitivity_analysis function - functionality available in main sensitivity_analysis

# ==============================
# Influence Analysis
# ==============================

@dataclass
class _PredictorRow:
    """describing one predictor row for a target j."""
    row_idx: int        # row in F/varF
    inputs: np.ndarray  # indices of input nodes (K_ij)
    prob: float         # selection probability for this predictor of target j


def _ia_build_predictors(network) -> List[List[_PredictorRow]]:
    """
    Decode predictors for each target j from ProbabilisticBN:
      - rows for j: cumsum[j] .. cumsum[j+1]-1
      - inputs: varF[row, :K[row]] (ignore -1)
      - prob (relative k within target j): cij[j, k]   (robustly normalized)
    """
    N = int(network.N)
    varF = np.asarray(network.varF)
    K = np.asarray(network.K)
    cij = np.asarray(network.cij)
    cumsum = np.asarray(network.cumsum)

    preds_per_j: List[List[_PredictorRow]] = []
    for j in range(N):
        rows = list(range(int(cumsum[j]), int(cumsum[j + 1])))
        probs = cij[j, :len(rows)].astype(float)
        s = probs[probs >= 0].sum()
        if s <= 1e-12:
            probs = np.ones_like(probs, dtype=float) / max(1, len(probs))
        else:
            probs = probs / s

        preds: List[_PredictorRow] = []
        for rel_k, r in enumerate(rows):
            Kj = int(K[r])
            if Kj > 0:
                inputs = varF[r, :Kj].astype(int)
                inputs = inputs[inputs >= 0]
            else:
                inputs = np.array([], dtype=int)
            preds.append(_PredictorRow(row_idx=int(r), inputs=inputs, prob=float(probs[rel_k])))
        preds_per_j.append(preds)
    return preds_per_j


def _ia_eval_predictor_row(network, row_idx: int, state: np.ndarray) -> int:
    """Evaluate F[row_idx] on a single binary state vector."""
    K = int(network.K[row_idx])
    if K == 0:
        return int(network.F[row_idx, 0])
    inputs = network.varF[row_idx, :K].astype(int)
    bits = state[inputs].astype(int)
    # inputs are ordered MSB first
    idx = 0
    for b in bits:
        idx = (idx << 1) | int(b)
    return int(network.F[row_idx, idx])


def _ia_eval_predictor_on_samples(network, row_idx: int, samples: np.ndarray) -> np.ndarray:
    """Vectorized evaluation of a predictor over many samples (loop over rows)."""
    S = samples.shape[0]
    out = np.empty(S, dtype=np.int8)
    for s in range(S):
        out[s] = _ia_eval_predictor_row(network, row_idx, samples[s])
    return out


def _ia_mismatch_rate_on_flipped(network, row_idx: int, samples: np.ndarray,
                                 flip_col: int, batch_size: int = 1024) -> float:
    """Compute P[f(X) != f(X^i)] by flipping column i in batches for memory efficiency."""
    S = samples.shape[0]
    mismatches = 0
    start = 0
    while start < S:
        end = min(start + batch_size, S)
        base_batch = samples[start:end]
        flip_batch = base_batch.copy()
        flip_batch[:, flip_col] ^= 1
        f_base = _ia_eval_predictor_on_samples(network, row_idx, base_batch)
        f_flip = _ia_eval_predictor_on_samples(network, row_idx, flip_batch)
        mismatches += int(np.count_nonzero(f_flip != f_base))
        start = end
    return mismatches / float(S)


def _ia_sample_states(network,
                      method: str = 'tsmc',
                      samples: int = 20000,
                      burn_in: int = 5000,
                      thin: int = 10,
                      epsilon: float = 0.001,
                      p_noise: float = 0.0,
                      seed: int = 42) -> np.ndarray:
    """
    Use SteadyStateCalculator to generate (approx.) steady-state samples.
    For TSMC, we request a long trajectory and then thin; for MC we use network.update/_noise.
    """
    N = int(network.N)
    rng = np.random.default_rng(seed)

    if method.lower() == 'tsmc':
        calc = SteadyStateCalculator(network)
        total_steps = burn_in + samples * thin
        y = calc.compute_stationary_tsmc(
            epsilon=epsilon,
            p_noise=0.0,
            initial_nsteps=total_steps,
            max_iterations=1,
            verbose=False,
            seed=seed
        )
        if y.ndim != 2 or y.shape[1] != N:
            raise RuntimeError("Unexpected trajectory shape from compute_stationary_tsmc")
        xs = y[burn_in::thin]
        if xs.shape[0] > samples:
            xs = xs[:samples]
        return xs.astype(np.int8)

    elif method.lower() in ('monte_carlo', 'mc'):
        init = (rng.random(N) > 0.5).astype(np.int8)
        network.setInitialValues(init)
        if p_noise > 0:
            network.update_noise(p_noise, iterations=burn_in)
        else:
            network.update(iterations=burn_in)

        collected = []
        for _ in range(samples):
            if p_noise > 0:
                network.update_noise(p_noise, iterations=thin)
            else:
                network.update(iterations=thin)
            collected.append(network.nodes.copy())
        return np.vstack(collected).astype(np.int8)

    else:
        raise ValueError("Unknown sampling method. Use 'tsmc' or 'monte_carlo'.")



def influence_analysis(
    network,
    output_node: Optional[Union[str, int]] = None,
    config: Optional[Dict] = None,
    top_n: int = 10,
    only_inputs: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Perform influence analysis on a Probabilistic Boolean Network.
    
    This function calculates the influence of each node on others based on the 
    Shmulevich et al. (2002) definition:
        Inf(i -> j) = sum_k p_k^{(j)} * P_X[ f_k^{(j)}(X) != f_k^{(j)}(X^{(i)}) ]
    where X is sampled from the steady state distribution.
    
    Parameters:
    -----------
    network : ProbabilisticBN
        The PBN object to analyze
    output_node : str or int, optional
        If provided, returns a ranking of sources influencing this target.
        Accepts node name (from network.nodeDict) or 0-based index.
    config : dict, optional
        Configuration parameters for steady state sampling:
        - method: 'tsmc' or 'monte_carlo' (default: 'tsmc')
        - samples: Number of steady state samples (default: 20000)
        - burn_in: Burn-in steps for sampling (default: 5000)
        - thin: Thinning interval for sampling (default: 10)
        - epsilon: Convergence threshold for TSMC (default: 0.001)
        - p_noise: Noise probability for Monte Carlo (default: 0.0)
        - seed: Random seed for reproducibility (default: 9)
        - batch_size: Batch size for memory efficiency (default: 1024)
    top_n : int
        Number of top sources to return for output_node (default: 10)
    only_inputs : bool
        If True, consider only inputs of j's predictors (faster).
        If False, consider all nodes as potential sources (default: True)
    verbose : bool
        Whether to print progress information and results (default: True)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'influence_matrix': np.ndarray (N x N), influence from row i to column j
        - 'influence_df': DataFrame with columns ['source', 'target', 'influence']
        - 'sensitivity_df': DataFrame with columns ['target', 'sensitivity']
        - 'ranking_df': DataFrame with top N sources (if output_node specified)
        - 'node_names': List of node names in order
        
    References:
    -----------
    Shmulevich, I., Dougherty, E. R., Kim, S., & Zhang, W. (2002). 
    Probabilistic Boolean Networks: a rule-based uncertainty model for gene regulatory networks. 
    Bioinformatics, 18(2), 261-274.
    """
    start_time = time.time()
    
    # Set default configuration
    cfg = {
        'method': 'tsmc',
        'samples': 20000,
        'burn_in': 5000,
        'thin': 10,
        'epsilon': 0.001,
        'p_noise': 0.0,
        'seed': 9,
        'batch_size': 1024,
    }
    if config:
        cfg.update(cfg)
    
    if verbose:
        print("="*60)
        print("PBN INFLUENCE ANALYSIS")
        print("="*60)
        print(f"Network size: {network.N} nodes")
        print(f"Sampling method: {cfg['method'].upper()}")
        print(f"Steady state samples: {cfg['samples']}")
    
    N = int(network.N)
    
    # Get node names
    if hasattr(network, 'nodeDict') and isinstance(network.nodeDict, dict):
        idx2name = {v: k for k, v in network.nodeDict.items()}
    else:
        idx2name = {i: f'x{i}' for i in range(N)}
    node_names = [idx2name[i] for i in range(N)]

    # Build predictors for each target
    preds_per_j = _ia_build_predictors(network)

    # Sample steady states
    if verbose:
        print(f"\nGenerating {cfg['samples']} steady state samples...")
    samples = _ia_sample_states(network, **cfg)
    S = samples.shape[0]

    # Precompute f_k(X) for all predictors
    if verbose:
        print("Precomputing predictor evaluations...")
    fk_vals: Dict[Tuple[int, int], np.ndarray] = {}
    for j, preds in enumerate(preds_per_j):
        for k, pr in enumerate(preds):
            fk_vals[(j, k)] = _ia_eval_predictor_on_samples(network, pr.row_idx, samples)

    # Calculate influence matrix
    if verbose:
        print("Calculating influence matrix...")
    Inf = np.zeros((N, N), dtype=float)
    for j, preds in enumerate(preds_per_j):
        if not preds:
            continue
        cand_i = sorted({int(x) for pr in preds for x in pr.inputs}) if only_inputs else list(range(N))
        for i in cand_i:
            total_change = 0.0
            for k, pr in enumerate(preds):
                change_rate = _ia_mismatch_rate_on_flipped(
                    network, pr.row_idx, samples, flip_col=i, batch_size=cfg['batch_size']
                )
                total_change += pr.prob * change_rate
            Inf[i, j] = total_change

    # Calculate sensitivity per target (incoming influence)
    sens = Inf.sum(axis=0)
    sensitivity_df = (
        pd.DataFrame({'target': node_names, 'sensitivity': sens})
        .sort_values('sensitivity', ascending=False)
        .reset_index(drop=True)
    )

    # Create long-form influence table
    infl_rows = [(node_names[i], node_names[j], Inf[i, j])
                 for j in range(N) for i in range(N) if Inf[i, j] > 0]
    influence_df = (
        pd.DataFrame(infl_rows, columns=['source', 'target', 'influence'])
        .sort_values('influence', ascending=False)
        .reset_index(drop=True)
    )

    # Prepare output
    out = {
        'influence_matrix': Inf,
        'influence_df': influence_df,
        'sensitivity_df': sensitivity_df,
        'node_names': node_names
    }

    # optional ranking for specific output node
    if output_node is not None:
        if isinstance(output_node, int):
            j = int(output_node)
        else:
            if hasattr(network, 'nodeDict') and output_node in network.nodeDict:
                j = int(network.nodeDict[output_node])
            else:
                # fallback: exact name match
                rev = {v: k for k, v in idx2name.items()}
                j = int(rev[output_node]) if output_node in rev else None
        if j is None:
            raise KeyError(f"Unknown output_node: {output_node}")

        ranking_df = (
            pd.DataFrame({'source': node_names, 'influence': Inf[:, j]})
            .sort_values('influence', ascending=False)
            .reset_index(drop=True)
        )
        out['ranking_df'] = ranking_df.head(top_n)

        if verbose:
            print("=" * 60)
            print(f"Top {top_n} sources influencing '{idx2name[j]}' (j={j}):")
            for _, row in out['ranking_df'].iterrows():
                print(f"  {row['source']}: {row['influence']:.6f}")
            print("=" * 60)

    # Print summary statistics
    if verbose:
        analysis_time = time.time() - start_time
        print(f"\nInfluence matrix summary:")
        print(f"  - Min influence: {Inf.min():.6f}")
        print(f"  - Max influence: {Inf.max():.6f}")
        print(f"  - Mean influence: {Inf.mean():.6f}")
        print(f"  - Analysis time: {analysis_time:.2f} seconds")
        print("="*60)
    
    return out

# ==============================
# MSE Sensitivity Analysis
# ==============================

def mse_sensitivity_analysis(network, experiments, config=None, top_n=5, verbose=True):
    """
    Perform MSE sensitivity analysis to identify which nodes have the largest impact on model fitting quality.
    
    This function analyzes how changes in node probabilities affect the Mean Squared Error (MSE) 
    between model predictions and experimental data. It's useful for understanding which nodes
    are most critical for maintaining the model's fit to experimental data.
    
    Parameters:
    -----------
    network : ProbabilisticBN
        The optimized PBN object to analyze
    experiments : str or list
        Path to experiments CSV or list of experiment dictionaries
    config : dict, optional
        Configuration including steady state parameters
    top_n : int
        Number of top sensitive nodes to return
    verbose : bool
        Whether to print detailed progress information
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'top_nodes': List of top N sensitive node names
        - 'sensitivity_df': DataFrame with all sensitivity results
        - 'baseline_mse': Baseline MSE with original parameters
        - 'experiment_errors': Per-experiment error breakdown
    """
    start_time = time.time()
    
    # Initialize analyzer
    analyzer = MSESensitivityAnalyzer(network, experiments, config, verbose)
    
    if verbose:
        print("="*60)
        print("PBN MSE SENSITIVITY ANALYSIS")
        print("="*60)
    
    # Run MSE sensitivity analysis
    sensitivity_df = analyzer.run_analysis()
    
    # Get top nodes
    top_nodes = analyzer.get_top_sensitive_nodes(sensitivity_df, top_n)
    
    analysis_time = time.time() - start_time
    
    if verbose:
        print(f"\nTotal analysis time: {analysis_time:.2f} seconds")
        print("="*60)
    
    return {
        'top_nodes': top_nodes,
        'sensitivity_df': sensitivity_df,
        'baseline_mse': analyzer.baseline_mse,
        'experiment_errors': analyzer.experiment_errors
    }


class MSESensitivityAnalyzer:
    """
    Performs MSE sensitivity analysis on PBN to identify nodes critical for model fitting.
    """
    
    def __init__(self, network, experiments, config=None, verbose=True):
        self.pbn = network
        self.verbose = verbose
        
        # Load experiments if path provided
        if isinstance(experiments, str):
            from KGBN.experiment_data import ExperimentData
            self.experiments = ExperimentData.load_from_csv(experiments)
        else:
            self.experiments = experiments
            
        # Set up configuration
        self.config = self._default_config()
        if config:
            self.config.update(config)
            
        # Initialize simulation evaluator for steady state calculations
        from KGBN.simulation_evaluator import SimulationEvaluator
        self.evaluator = SimulationEvaluator(
            self.pbn, 
            self.experiments, 
            self.config
        )
        
        # Extract measured nodes from experiments
        self.measured_nodes = self._get_measured_nodes()
        
        # Determine which nodes can be analyzed (have multiple functions)
        self.analyzable_nodes = self._get_analyzable_nodes()
        
        # Store original Cij for faster restoration
        self.original_cij = self.pbn.cij.copy()
        
        # Calculate baseline MSE
        self.baseline_mse = self._calculate_baseline_mse()
        self.experiment_errors = self._calculate_experiment_errors()
    
    def _default_config(self):
        """Default configuration"""
        return {
            'seed': 9,                            # Global seed for reproducibility
            'perturbation_magnitude': 0.1,         # How much to perturb probabilities
            'steady_state': {
                'method': 'tsmc',
                'tsmc_params': {
                    'epsilon': 0.01,        
                    'r': 0.05,               
                    's': 0.90,              
                    'p_mir': 0.001,
                    'initial_nsteps': 100,       
                    'max_iterations': 1000,       
                },
                'monte_carlo_params': {
                    'n_runs': 2,              
                    'n_steps': 5000,           
                    'p_noise': 0.05
                }
            }
        }
    
    def _get_measured_nodes(self):
        """Extract all measured nodes from experiments"""
        measured = set()
        for exp in self.experiments:
            measured.update(exp['measurements'].keys())
        return list(measured)
    
    def _get_analyzable_nodes(self):
        """Get nodes that can be analyzed (have variable functions)"""
        analyzable = []
        for node_name, node_idx in self.pbn.nodeDict.items():
            # Include all nodes with multiple functions (including measured ones)
            if self.pbn.nf[node_idx] > 1:
                analyzable.append((node_name, node_idx))
        return analyzable
    
    def _calculate_baseline_mse(self):
        """Calculate baseline MSE with original parameters"""
        total_sse = 0
        for experiment in self.experiments:
            try:
                predicted = self.evaluator._simulate_experiment(experiment)
                sse = self.evaluator._calculate_sse(predicted, experiment['measurements'])
                total_sse += sse
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Baseline simulation failed for experiment: {e}")
                total_sse += 1e10  # Large penalty for failed simulations
        
        return total_sse / len(self.experiments)
    
    def _calculate_experiment_errors(self):
        """Calculate per-experiment errors for baseline"""
        errors = {}
        for i, experiment in enumerate(self.experiments):
            try:
                predicted = self.evaluator._simulate_experiment(experiment)
                sse = self.evaluator._calculate_sse(predicted, experiment['measurements'])
                errors[f"Experiment_{i+1}"] = sse
            except Exception:
                errors[f"Experiment_{i+1}"] = 1e10
        return errors
    
    def run_analysis(self):
        """Run the MSE sensitivity analysis"""
        if self.verbose:
            print(f"\nNetwork information:")
            print(f"  - Total nodes: {len(self.pbn.nodeDict)}")
            print(f"  - Measured nodes: {len(self.measured_nodes)} ({', '.join(self.measured_nodes)})")
            print(f"  - Analyzable nodes: {len(self.analyzable_nodes)}")
            print(f"  - Experiments: {len(self.experiments)}")
            print(f"  - Baseline MSE: {self.baseline_mse:.6f}")
        
        # Run one-at-a-time sensitivity analysis
        return self._run_oat_analysis()
    
    def _run_oat_analysis(self):
        """Run one-at-a-time MSE sensitivity analysis"""
        if self.verbose:
            print(f"\nRunning One-at-a-Time MSE sensitivity analysis...")
            print(f"  - Perturbation magnitude: {self.config.get('perturbation_magnitude', 0.3)}")
        
        results = []
        
        # Test each analyzable node
        for i, (node_name, node_idx) in enumerate(self.analyzable_nodes):
            if self.verbose and i % 5 == 0:
                print(f"  Progress: {i}/{len(self.analyzable_nodes)}")
            
            # Test with perturbed probabilities
            test_cij = self._create_perturbed_cij(node_idx)
            
            # Calculate MSE with perturbed parameters
            perturbed_mse = self._calculate_perturbed_mse(test_cij)
            
            # Calculate sensitivity metrics
            mse_change = perturbed_mse - self.baseline_mse
            mse_change_relative = mse_change / self.baseline_mse if self.baseline_mse > 0 else 0
            
            # Store results for each measured node
            for measured_node in self.measured_nodes:
                results.append({
                    'node': node_name,
                    'measured_node': measured_node,
                    'baseline_mse': self.baseline_mse,
                    'perturbed_mse': perturbed_mse,
                    'mse_change': mse_change,
                    'mse_change_relative': mse_change_relative,
                    'sensitivity_score': abs(mse_change_relative)
                })
        
        # Restore original parameters
        self.evaluator._update_pbn_parameters(self.original_cij)
        
        return pd.DataFrame(results)
    
    def _create_perturbed_cij(self, node_idx):
        """Create perturbed Cij matrix for a specific node"""
        cij_matrix = self.original_cij.copy()
        n_functions = self.pbn.nf[node_idx]
        perturbation = self.config.get('perturbation_magnitude', 0.1)

        # Copy the original probabilities for this node
        orig_probs = cij_matrix[node_idx, :n_functions].copy()
        new_probs = orig_probs.copy()
        # Change the first function's probability by perturbation magnitude
        if new_probs[0] + perturbation > 1:
            perturbation = 1 - new_probs[0]
            new_probs[0] = 1
        elif new_probs[0] + perturbation < 0:
            perturbation = -new_probs[0]
            new_probs[0] = 0
        else:
            new_probs[0] += perturbation

        # To keep the sum to 1, subtract the perturbation from the other functions proportionally
        if n_functions > 1:
            rest_sum = np.sum(orig_probs[1:])
            if rest_sum > 0:
                new_probs[1:] -= perturbation * (orig_probs[1:] / rest_sum)
            else:
                # If all other probabilities are zero, just distribute equally
                new_probs[1:] -= perturbation / (n_functions - 1)
        # Ensure no negative probabilities
        new_probs = np.maximum(new_probs, 1e-10)
        # Renormalize to sum to 1
        new_probs = new_probs / np.sum(new_probs)
        # Update Cij matrix
        for j in range(len(cij_matrix[node_idx])):
            if j < n_functions:
                cij_matrix[node_idx, j] = new_probs[j]
            else:
                cij_matrix[node_idx, j] = -1

        return cij_matrix
    
    def _calculate_perturbed_mse(self, cij_matrix):
        """Calculate MSE with perturbed parameters"""
        # Update PBN parameters
        self.evaluator._update_pbn_parameters(cij_matrix)
        
        # Calculate total SSE
        total_sse = 0
        for experiment in self.experiments:
            try:
                predicted = self.evaluator._simulate_experiment(experiment)
                sse = self.evaluator._calculate_sse(predicted, experiment['measurements'])
                total_sse += sse
            except Exception:
                total_sse += 1e10  # Large penalty for failed simulations
        
        return total_sse / len(self.experiments)
    
    def get_top_sensitive_nodes(self, sensitivity_df, top_n=5):
        """Extract top N sensitive nodes from results"""
        if self.verbose:
            print(f"\nIdentifying top {top_n} sensitive nodes...")
        
        # Aggregate sensitivity across all measured nodes
        node_sensitivity = sensitivity_df.groupby('node')['sensitivity_score'].agg(['mean', 'max', 'std'])
        
        # Sort by mean sensitivity
        node_sensitivity = node_sensitivity.sort_values('mean', ascending=False)
        
        # Get top N
        top_nodes = node_sensitivity.head(top_n).index.tolist()
        
        if self.verbose:
            print(f"\nTop {top_n} sensitive nodes (by average sensitivity score):")
            print("-" * 60)
            for i, node in enumerate(top_nodes):
                stats = node_sensitivity.loc[node]
                print(f"{i+1}. {node:15s} - Avg: {stats['mean']:.4f}, Max: {stats['max']:.4f}, Std: {stats['std']:.4f}")
            
            # Show MSE change details for top nodes
            print(f"\nMSE change details for top nodes:")
            print("-" * 60)
            for node in top_nodes[:min(3, len(top_nodes))]:  # Show details for top 3
                node_data = sensitivity_df[sensitivity_df['node'] == node]
                avg_mse_change = node_data['mse_change'].mean()
                avg_relative_change = node_data['mse_change_relative'].mean()
                print(f"\n{node}:")
                print(f"  Average MSE change: {avg_mse_change:+.6f}")
                print(f"  Average relative change: {avg_relative_change:+.2%}")
        
        return top_nodes

def plot_influence_results(influence_df, sensitivity_df, top_n=20, save_path=None):
    """
    Create visualization of influence analysis results.
    
    Parameters:
    -----------
    influence_df : pd.DataFrame
        Results from influence analysis with columns ['source', 'target', 'influence']
    sensitivity_df : pd.DataFrame
        Sensitivity results with columns ['target', 'sensitivity']
    top_n : int
        Number of top nodes to display
    save_path : str, optional
        Path to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Top sources by total outgoing influence
    source_influence = influence_df.groupby('source')['influence'].sum().sort_values(ascending=False).head(top_n)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(source_influence)))
    bars = ax1.barh(range(len(source_influence)), source_influence.values, color=colors)
    ax1.set_yticks(range(len(source_influence)))
    ax1.set_yticklabels(source_influence.index)
    ax1.set_xlabel('Total Outgoing Influence')
    ax1.set_title(f'Top {top_n} Sources by Total Influence')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, source_influence.values)):
        ax1.text(value + 0.001, i, f'{value:.3f}', va='center', fontsize=8)
    
    # Plot 2: Top targets by sensitivity (incoming influence)
    top_targets = sensitivity_df.head(top_n)
    
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_targets)))
    bars = ax2.barh(range(len(top_targets)), top_targets['sensitivity'].values, color=colors)
    ax2.set_yticks(range(len(top_targets)))
    ax2.set_yticklabels(top_targets['target'])
    ax2.set_xlabel('Sensitivity (Incoming Influence)')
    ax2.set_title(f'Top {top_n} Targets by Sensitivity')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_targets['sensitivity'].values)):
        ax2.text(value + 0.001, i, f'{value:.3f}', va='center', fontsize=8)
    
    # Plot 3: Heatmap of top influences
    top_sources = source_influence.head(10).index
    top_targets_names = top_targets.head(10)['target'].values
    
    # Filter influence data for top sources and targets
    heatmap_data = influence_df[
        (influence_df['source'].isin(top_sources)) & 
        (influence_df['target'].isin(top_targets_names))
    ].pivot(index='source', columns='target', values='influence')
    
    # Fill NaN values with 0
    heatmap_data = heatmap_data.fillna(0)
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Influence'}, ax=ax3)
    ax3.set_title('Influence Matrix (Top Sources vs Top Targets)')
    ax3.set_xlabel('Target Node')
    ax3.set_ylabel('Source Node')
    
    # Plot 4: Distribution of influence values
    influence_values = influence_df['influence'].values
    ax4.hist(influence_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Influence Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Influence Values')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics
    mean_inf = np.mean(influence_values)
    median_inf = np.median(influence_values)
    ax4.axvline(mean_inf, color='red', linestyle='--', label=f'Mean: {mean_inf:.3f}')
    ax4.axvline(median_inf, color='orange', linestyle='--', label=f'Median: {median_inf:.3f}')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_mse_sensitivity_results(sensitivity_df, baseline_mse, top_n=20, save_path=None):
    """
    Create visualization of MSE sensitivity analysis results.
    
    Parameters:
    -----------
    sensitivity_df : pd.DataFrame
        Results from MSE sensitivity analysis
    baseline_mse : float
        Baseline MSE with original parameters
    top_n : int
        Number of top nodes to display
    save_path : str, optional
        Path to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    
    # MSE change distribution
    mse_changes = sensitivity_df.groupby('node')['mse_change'].mean().sort_values(ascending=False).head(top_n)
    
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(mse_changes)))
    bars = ax1.barh(range(len(mse_changes)), mse_changes.values, color=colors)
    ax1.set_yticks(range(len(mse_changes)))
    ax1.set_yticklabels(mse_changes.index)
    ax1.set_xlabel('Average MSE Change')
    ax1.set_title(f'Top {top_n} Nodes by MSE Change')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, mse_changes.values)):
        ax1.text(value + (0.001 if value >= 0 else -0.001), i, f'{value:+.4f}', 
                va='center', fontsize=8, ha='left' if value >= 0 else 'right')
    
    # Relative MSE change vs baseline
    relative_changes = sensitivity_df.groupby('node')['mse_change_relative'].mean().sort_values(ascending=False).head(top_n)
        
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(relative_changes)))
    bars = ax2.barh(range(len(relative_changes)), relative_changes.values, color=colors)
    ax2.set_yticks(range(len(relative_changes)))
    ax2.set_yticklabels(relative_changes.index)
    ax2.set_xlabel('Average Relative MSE Change')
    ax2.set_title(f'Top {top_n} Nodes by Relative MSE Change')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (bar, value) in enumerate(zip(bars, relative_changes.values)):
        ax2.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:+.1%}', 
                va='center', fontsize=8, ha='left' if value >= 0 else 'right')
    
    # Add baseline MSE info
    fig.suptitle(f'MSE Sensitivity Analysis Results (Baseline MSE: {baseline_mse:.6f})', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()