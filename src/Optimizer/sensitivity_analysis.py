import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from SALib.sample import saltelli, morris as morris_sampler
from SALib.analyze import sobol, morris
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings

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
            from Optimizer.experiment_data import ExperimentData
            self.experiments = ExperimentData.load_from_csv(experiments)
        else:
            self.experiments = experiments
            
        # Set up configuration
        self.config = self._default_config()
        if config:
            self.config.update(config)
            
        # Initialize simulation evaluator for steady state calculations
        from Optimizer.simulation_evaluator import SimulationEvaluator
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
    from Optimizer.simulation_evaluator import SimulationEvaluator
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

# Utility functions remain the same
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


def simple_sensitivity_analysis(network, experiments, config=None, top_n=5):
    """
    Simplified sensitivity analysis without parallelization for debugging.
    Uses a basic one-at-a-time approach.
    
    Parameters:
    -----------
    network : ProbabilisticBN
        The PBN object to analyze
    experiments : str or list
        Path to experiments CSV or list of experiment dictionaries
    config : dict, optional
        Configuration (steady state parameters)
    top_n : int
        Number of top sensitive nodes to return
        
    Returns:
    --------
    dict
        Dictionary with top_nodes and sensitivity_df
    """
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    print("="*60)
    print("SIMPLE SENSITIVITY ANALYSIS (Non-parallel)")
    print("="*60)
    
    # Load experiments if needed
    if isinstance(experiments, str):
        from Optimizer.experiment_data import ExperimentData
        experiments = ExperimentData.load_from_csv(experiments)
    
    # Get default config
    default_config = {
        'steady_state': {
            'method': 'tsmc',
            'tsmc_params': {
                'epsilon': 0.02,
                'r': 0.1,
                's': 0.85,
                'initial_nsteps': 20,
                'max_iterations': 100
            }
        }
    }
    if config:
        default_config.update(config)
    
    # Initialize evaluator
    from Optimizer.simulation_evaluator import SimulationEvaluator
    evaluator = SimulationEvaluator(network, experiments, default_config)
    
    # Get measured nodes
    measured_nodes = set()
    for exp in experiments:
        measured_nodes.update(exp['measurements'].keys())
    measured_nodes = list(measured_nodes)
    
    # Get analyzable nodes
    analyzable_nodes = []
    for node_name, node_idx in network.nodeDict.items():
        if node_name not in measured_nodes and network.nf[node_idx] > 1:
            analyzable_nodes.append((node_name, node_idx))
    
    print(f"\nAnalyzing {len(analyzable_nodes)} nodes")
    print(f"Measured nodes: {', '.join(measured_nodes)}")
    
    # Store original Cij
    original_cij = network.cij.copy()
    
    # Run simple OAT analysis
    results = []
    
    # Get baseline
    print("\nCalculating baseline...")
    baseline_values = {}
    for exp_idx, experiment in enumerate(experiments):
        evaluator._update_pbn_parameters(original_cij)
        steady_state = evaluator._simulate_experiment(experiment)
        for measured_node in measured_nodes:
            key = (exp_idx, measured_node)
            baseline_values[key] = steady_state[network.nodeDict[measured_node]]
    
    # Test each node
    print("\nTesting nodes...")
    for i, (node_name, node_idx) in enumerate(analyzable_nodes):
        if i % 5 == 0:
            print(f"  Progress: {i}/{len(analyzable_nodes)}")
        
        # Test with modified probability
        test_cij = original_cij.copy()
        n_functions = network.nf[node_idx]
        
        # Create perturbed probabilities (emphasize first function)
        new_probs = np.ones(n_functions) * 0.1 / (n_functions - 1)
        new_probs[0] = 0.9
        
        # Update Cij matrix
        for j in range(len(test_cij[node_idx])):
            if j < n_functions:
                test_cij[node_idx, j] = new_probs[j]
            else:
                test_cij[node_idx, j] = -1
        
        # Calculate effects
        effects = []
        for exp_idx, experiment in enumerate(experiments):
            evaluator._update_pbn_parameters(test_cij)
            steady_state = evaluator._simulate_experiment(experiment)
            
            for measured_node in measured_nodes:
                baseline = baseline_values[(exp_idx, measured_node)]
                perturbed = steady_state[network.nodeDict[measured_node]]
                effect = abs(perturbed - baseline)
                effects.append(effect)
        
        # Store average effect
        avg_effect = np.mean(effects)
        for measured_node in measured_nodes:
            results.append({
                'node': node_name,
                'measured_node': measured_node,
                'mu_star': avg_effect,
                'mu': avg_effect,
                'sigma': np.std(effects)
            })
    
    # Restore original
    evaluator._update_pbn_parameters(original_cij)
    
    # Create DataFrame
    import pandas as pd
    sensitivity_df = pd.DataFrame(results)
    
    # Get top nodes
    node_sensitivity = sensitivity_df.groupby('node')['mu_star'].mean().sort_values(ascending=False)
    top_nodes = node_sensitivity.head(top_n).index.tolist()
    
    print(f"\nTop {top_n} sensitive nodes:")
    for i, node in enumerate(top_nodes):
        print(f"{i+1}. {node}: {node_sensitivity[node]:.4f}")
    
    print("="*60)
    
    return {
        'top_nodes': top_nodes,
        'sensitivity_df': sensitivity_df
    }