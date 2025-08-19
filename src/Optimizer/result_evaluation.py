import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional
import pandas as pd
from .simulation_evaluator import SimulationEvaluator
from .experiment_data import ExperimentData

class ResultEvaluator:
    """
    Evaluate optimization results by comparing simulation output with experimental data.
    
    This provides tools to assess the quality of optimized models by:
    1. Simulating the optimized model on experimental conditions
    2. Comparing simulation results with experimental measurements
    3. Calculating correlation and other statistical metrics
    4. Generating visualization plots
    """
    
    def __init__(self, optimizer_result, parameter_optimizer):
        """
        Initialize the result evaluator.
        
        Parameters:
        -----------
        optimizer_result : OptimizeResult
            The optimization result to evaluate
        parameter_optimizer : ParameterOptimizer
            The parameter optimizer instance containing the experiments and configuration for simulation
        """
        self.result = optimizer_result
        self.optimizer = parameter_optimizer
        self.pbn = parameter_optimizer.pbn
        self.evaluator = parameter_optimizer.evaluator
        self.experiments = parameter_optimizer.evaluator.experiments
        
        # Store simulation results
        self.simulation_results = None
        self.evaluation_metrics = None
        
    def simulate_optimized_model(self) -> Dict:
        """
        Simulate the optimized model on all experimental conditions.
        
        Returns:
        --------
        Dict
            Dictionary containing simulation results for each experiment
        """
        print("Simulating optimized model on all experimental conditions...")
        
        # Set global seed for reproducibility
        global_seed = self.optimizer.config.get('seed', 9)
        np.random.seed(global_seed)
        
        simulation_results = {
            'experiments': [],
            'predictions': [],
            'measurements': [],
            'experiment_ids': [],
            'measured_nodes': [],
            'predicted_values': [],
            'measured_values': []
        }
        
        # Ensure the PBN has the optimized parameters
        if hasattr(self.result, 'x') and self.result.x is not None:
            cij_matrix = self.evaluator._vector_to_cij_matrix(self.result.x)
            self.evaluator._update_pbn_parameters(cij_matrix)
        
        for i, experiment in enumerate(self.experiments):
            try:
                # Simulate experiment
                predicted_steady_state = self.evaluator._simulate_experiment(experiment)
                
                # Extract predictions for measured nodes
                exp_predictions = {}
                exp_measurements = {}
                
                for node_name, measured_value in experiment['measurements'].items():
                    if node_name in self.pbn.nodeDict:
                        node_idx = self.pbn.nodeDict[node_name]
                        predicted_value = predicted_steady_state[node_idx]
                        
                        exp_predictions[node_name] = predicted_value
                        exp_measurements[node_name] = measured_value
                        
                        # Store for correlation analysis
                        simulation_results['predicted_values'].append(predicted_value)
                        simulation_results['measured_values'].append(measured_value)
                        simulation_results['measured_nodes'].append(node_name)
                        simulation_results['experiment_ids'].append(experiment.get('id', i+1))
                
                simulation_results['experiments'].append(experiment)
                simulation_results['predictions'].append(exp_predictions)
                simulation_results['measurements'].append(exp_measurements)
                
                print(f"  Experiment {i+1}: {len(exp_predictions)} nodes simulated")
                
            except Exception as e:
                print(f"  Warning: Failed to simulate experiment {i+1}: {str(e)}")
                
        self.simulation_results = simulation_results
        
        print(f"Simulation completed: {len(simulation_results['predicted_values'])} data points")
        return simulation_results
    
    def calculate_evaluation_metrics(self) -> Dict:
        """
        Calculate evaluation metrics comparing simulation results with experimental data.
        
        Returns:
        --------
        Dict
            Dictionary containing various evaluation metrics
        """
        if self.simulation_results is None:
            self.simulate_optimized_model()
        
        predicted = np.array(self.simulation_results['predicted_values'])
        measured = np.array(self.simulation_results['measured_values'])
        
        if len(predicted) == 0 or len(measured) == 0:
            print("Warning: No data available for evaluation")
            return {}
        
        # Calculate correlation
        correlation, p_value = pearsonr(predicted, measured)
        
        # Calculate other metrics
        mse = np.mean((predicted - measured) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predicted - measured))
        
        # R-squared
        ss_res = np.sum((measured - predicted) ** 2)
        ss_tot = np.sum((measured - np.mean(measured)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate per-node metrics
        node_metrics = {}
        nodes = set(self.simulation_results['measured_nodes'])
        
        for node in nodes:
            node_indices = [i for i, n in enumerate(self.simulation_results['measured_nodes']) if n == node]
            node_pred = predicted[node_indices]
            node_meas = measured[node_indices]
            
            if len(node_pred) > 1:
                node_corr, node_p = pearsonr(node_pred, node_meas)
                node_mse = np.mean((node_pred - node_meas) ** 2)
                node_mae = np.mean(np.abs(node_pred - node_meas))
                
                node_metrics[node] = {
                    'correlation': node_corr,
                    'p_value': node_p,
                    'mse': node_mse,
                    'mae': node_mae,
                    'n_points': len(node_pred)
                }
        
        metrics = {
            'overall': {
                'correlation': correlation,
                'p_value': p_value,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r_squared': r_squared,
                'n_points': len(predicted)
            },
            'per_node': node_metrics,
            'optimization_result': {
                'final_mse': self.result.fun,
                'success': self.result.success,
                'iterations': self.result.nit,
                'function_evaluations': self.result.nfev
            }
        }
        
        self.evaluation_metrics = metrics
        return metrics
    
    def plot_prediction_vs_experimental(self, save_path: Optional[str] = None, 
                                      show_confidence_interval: bool = False,
                                      show_experiment_ids: bool = False,
                                      figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Create a scatter plot comparing predicted vs experimental values.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, displays the plot.
        show_confidence_interval : bool, default=True
            Whether to show confidence intervals
        show_experiment_ids : bool, default=False
            Whether to label points with experiment IDs
        figsize : Tuple[int, int], default=(8, 6)
            Figure size
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        if self.simulation_results is None:
            self.simulate_optimized_model()
        
        if self.evaluation_metrics is None:
            self.calculate_evaluation_metrics()
        
        predicted = np.array(self.simulation_results['predicted_values'])
        measured = np.array(self.simulation_results['measured_values'])
        nodes = self.simulation_results['measured_nodes']
        experiment_ids = self.simulation_results['experiment_ids']
        
        if len(predicted) == 0:
            print("No data available for plotting")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(measured, predicted, alpha=0.7, s=60, c='lightgreen')
        
        # Add experiment ID labels
        if show_experiment_ids:
            for i, (meas, pred, exp_id) in enumerate(zip(measured, predicted, experiment_ids)):
                ax.annotate(f'E{exp_id}', (meas, pred), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, alpha=0.7)
        
        # Perfect prediction line
        min_val = min(np.min(predicted), np.min(measured))
        max_val = max(np.max(predicted), np.max(measured))
        # ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
        #         linewidth=2, label='Perfect prediction')
        
        # Calculate regression line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(measured, predicted)
        
        # Create regression line
        x_reg = np.linspace(min_val, max_val, 100)
        y_reg = slope * x_reg + intercept
        ax.plot(x_reg, y_reg, 'g-', linewidth=2, alpha=0.8, label='Regression line')

        # Add confidence interval bands
        if show_confidence_interval:
            # Add confidence bands (approximate)
            residuals = predicted - (slope * measured + intercept)
            mse_residuals = np.mean(residuals**2)
            confidence_interval = 1.96 * np.sqrt(mse_residuals)  # 95% CI
            
            ax.fill_between(x_reg, y_reg - confidence_interval, y_reg + confidence_interval, 
                           alpha=0.2, color='green', label='95% Confidence interval')
        
        # Formatting
        ax.set_xlabel('Experimental Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        
        # key statistics
        metrics = self.evaluation_metrics['overall']
        final_mse = self.evaluation_metrics['optimization_result']['final_mse']
        
        # title = f'Predicted vs Experimental (r={metrics["correlation"]:.3f}, p={metrics["p_value"]:.3e}, MSE={final_mse:.6f})'
        title = f'Predicted vs Experimental (r={metrics["correlation"]:.3f}, p={metrics["p_value"]:.3e})'
        # title = f'Predicted vs Experimental'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Legend
        ax.legend()
        
        # Equal aspect ratio
        # ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(np.min(measured) - 0.05, np.max(measured) + 0.05)
        ax.set_ylim(np.min(predicted) - 0.05, np.max(predicted) + 0.05)
        
        # Tight layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_residuals(self, save_path: Optional[str] = None, 
                      show_experiment_ids: bool = False,
                      figsize: Tuple[int, int] = (9, 4)) -> plt.Figure:
        """
        Create residual plots to assess model fit quality.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        show_experiment_ids : bool, default=False
            Whether to label points with experiment IDs
        figsize : Tuple[int, int], default=(9, 4)
            Figure size in inches
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        if self.simulation_results is None:
            self.simulate_optimized_model()
        
        predicted = np.array(self.simulation_results['predicted_values'])
        measured = np.array(self.simulation_results['measured_values'])
        experiment_ids = self.simulation_results['experiment_ids']
        residuals = predicted - measured
        
        if len(predicted) == 0:
            print("No data available for plotting")
            return None
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Residuals vs Predicted
        ax1.scatter(predicted, residuals, alpha=0.7, s=60)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals (Predicted - Measured)')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True, alpha=0.3)
        
        # Add experiment ID labels if requested
        if show_experiment_ids:
            for pred, res, exp_id in zip(predicted, residuals, experiment_ids):
                ax1.annotate(f'E{exp_id}', (pred, res), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, alpha=0.7)
        
        # Histogram of residuals
        ax2.hist(residuals, bins=min(20, len(residuals)//3), alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        ax2.text(0.02, 0.98, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Residual plot saved to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def generate_evaluation_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the report as a text file
            
        Returns:
        --------
        str
            The evaluation report as a string
        """
        if self.evaluation_metrics is None:
            self.calculate_evaluation_metrics()
        
        report = []
        report.append("="*60)
        report.append("OPTIMIZATION RESULT EVALUATION REPORT")
        report.append("="*60)
        report.append("")
        
        # Overall metrics
        overall = self.evaluation_metrics['overall']
        report.append("OVERALL PERFORMANCE:")
        report.append("-" * 20)
        report.append(f"Pearson Correlation: {overall['correlation']:.4f}")
        report.append(f"P-value: {overall['p_value']:.6e}")
        report.append(f"R-squared: {overall['r_squared']:.4f}")
        report.append(f"Mean Squared Error: {overall['mse']:.6f}")
        report.append(f"Root Mean Squared Error: {overall['rmse']:.6f}")
        report.append(f"Mean Absolute Error: {overall['mae']:.6f}")
        report.append(f"Number of data points: {overall['n_points']}")
        report.append("")
        
        # Optimization details
        opt = self.evaluation_metrics['optimization_result']
        report.append("OPTIMIZATION DETAILS:")
        report.append("-" * 20)
        report.append(f"Final MSE: {opt['final_mse']:.6f}")
        report.append(f"Success: {opt['success']}")
        report.append(f"Iterations: {opt['iterations']}")
        report.append(f"Function evaluations: {opt['function_evaluations']}")
        report.append("")
        
        # Per-node metrics
        if self.evaluation_metrics['per_node']:
            report.append("PER-NODE PERFORMANCE:")
            report.append("-" * 20)
            for node, metrics in self.evaluation_metrics['per_node'].items():
                report.append(f"{node}:")
                report.append(f"  Correlation: {metrics['correlation']:.4f} (p = {metrics['p_value']:.4e})")
                report.append(f"  MSE: {metrics['mse']:.6f}")
                report.append(f"  MAE: {metrics['mae']:.6f}")
                report.append(f"  Data points: {metrics['n_points']}")
                report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Evaluation report saved to {save_path}")
        
        return report_text
    
    def export_results_to_csv(self, save_path: str):
        """
        Export detailed results to CSV for further analysis.
        
        Parameters:
        -----------
        save_path : str
            Path to save the CSV file
        """
        if self.simulation_results is None:
            self.simulate_optimized_model()
        
        # Create DataFrame with all results
        data = {
            'Experiment_ID': self.simulation_results['experiment_ids'],
            'Node': self.simulation_results['measured_nodes'],
            'Predicted_Value': self.simulation_results['predicted_values'],
            'Measured_Value': self.simulation_results['measured_values'],
            'Residual': np.array(self.simulation_results['predicted_values']) - np.array(self.simulation_results['measured_values']),
            'Absolute_Error': np.abs(np.array(self.simulation_results['predicted_values']) - np.array(self.simulation_results['measured_values']))
        }
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"Results exported to {save_path}")


def evaluate_optimization_result(optimizer_result, parameter_optimizer, 
                                output_dir: str = ".", 
                                plot_residuals: bool = True,
                                save: bool = True,
                                detailed: bool = False,
                                figsize: Tuple[int, int] = (8, 6),
                                show_confidence_interval: bool = False) -> ResultEvaluator:
    """
    Convenience function to perform a complete evaluation of optimization results.
    
    Parameters:
    -----------
    optimizer_result : OptimizeResult
        The optimization result
    parameter_optimizer : ParameterOptimizer
        The parameter optimizer instance
    plot_residuals: bool, default=True
        Whether to plot residuals
    output_dir : str, default="."
        Directory to save output files
    save : bool, default=True
        Whether to save plots and reports to files. If False, displays plots.
    detailed : bool, default=False
        Whether to label dots in plots with experiment IDs
    figsize : Tuple[int, int], default=(8, 6)
        Figure size in inches for the prediction vs experimental plot
    show_confidence_interval: bool, default=False
        Whether to show confidence interval bands
    Returns:
    --------
    ResultEvaluator
        The result evaluator instance
    """
    import os
    if save:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create evaluator
    evaluator = ResultEvaluator(optimizer_result, parameter_optimizer)
    
    # Calculate metrics
    evaluator.calculate_evaluation_metrics()
    
    # Generate plots
    if save:
        # Prediction vs experimental plot
        plot_path = os.path.join(output_dir, "prediction_vs_experimental.png")
        evaluator.plot_prediction_vs_experimental(save_path=plot_path, show_experiment_ids=detailed, figsize=figsize, show_confidence_interval=show_confidence_interval)

        if plot_residuals:
        # Residual plots
            residual_path = os.path.join(output_dir, "residual_analysis.png")
            evaluator.plot_residuals(save_path=residual_path, show_experiment_ids=detailed)
        
        # Generate report
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        evaluator.generate_evaluation_report(save_path=report_path)
        
        # Export CSV
        csv_path = os.path.join(output_dir, "detailed_results.csv")
        evaluator.export_results_to_csv(csv_path)
    else:
        # Display plots without saving
        evaluator.plot_prediction_vs_experimental(show_experiment_ids=detailed)
        if plot_residuals:
            evaluator.plot_residuals(show_experiment_ids=detailed)
        
        # Print report to console
        report = evaluator.generate_evaluation_report()
        print(report)
    
    return evaluator


def evaluate_pbn(pbn, experiments, output_dir: str = '.', generate_plots: bool = True, generate_report: bool = True, config: dict = None):
    """
    Evaluate a PBN directly against experiment data (list or CSV).

    Parameters:
    -----------
    pbn : ProbabilisticBN
        The PBN object to evaluate
    experiments : list or str
        List of experiment dicts or path to CSV file
    output_dir : str, default='.'
        Directory to save output files
    generate_plots : bool, default=True
        Whether to generate plots
    generate_report : bool, default=True
        Whether to generate evaluation report
    config : dict, optional
        Simulation configuration

    Returns:
    --------
    dict
        Dictionary with evaluation metrics and file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    from .experiment_data import ExperimentData
    from .simulation_evaluator import SimulationEvaluator
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr
    import matplotlib.pyplot as plt

    # Load experiments if needed
    if isinstance(experiments, str):
        experiments = ExperimentData.load_from_csv(experiments)
    # Validate experiments
    ExperimentData.validate_experiments(experiments, pbn.nodeDict)

    # Create evaluator
    evaluator = SimulationEvaluator(pbn, experiments, config)

    # Simulate all experiments
    simulation_results = {
        'experiments': [],
        'predictions': [],
        'measurements': [],
        'experiment_ids': [],
        'measured_nodes': [],
        'predicted_values': [],
        'measured_values': []
    }
    for i, experiment in enumerate(experiments):
        try:
            predicted_steady_state = evaluator._simulate_experiment(experiment)
            exp_predictions = {}
            exp_measurements = {}
            for node_name, measured_value in experiment['measurements'].items():
                if node_name in pbn.nodeDict:
                    node_idx = pbn.nodeDict[node_name]
                    predicted_value = predicted_steady_state[node_idx]
                    exp_predictions[node_name] = predicted_value
                    exp_measurements[node_name] = measured_value
                    simulation_results['predicted_values'].append(predicted_value)
                    simulation_results['measured_values'].append(measured_value)
                    simulation_results['measured_nodes'].append(node_name)
                    simulation_results['experiment_ids'].append(experiment.get('id', i+1))
            simulation_results['experiments'].append(experiment)
            simulation_results['predictions'].append(exp_predictions)
            simulation_results['measurements'].append(exp_measurements)
        except Exception as e:
            print(f"  Warning: Failed to simulate experiment {i+1}: {str(e)}")

    # Calculate metrics
    predicted = np.array(simulation_results['predicted_values'])
    measured = np.array(simulation_results['measured_values'])
    metrics = {}
    if len(predicted) > 0 and len(measured) > 0:
        correlation, p_value = pearsonr(predicted, measured)
        mse = np.mean((predicted - measured) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predicted - measured))
        ss_res = np.sum((measured - predicted) ** 2)
        ss_tot = np.sum((measured - np.mean(measured)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        # Per-node metrics
        node_metrics = {}
        nodes = set(simulation_results['measured_nodes'])
        for node in nodes:
            node_indices = [i for i, n in enumerate(simulation_results['measured_nodes']) if n == node]
            node_pred = predicted[node_indices]
            node_meas = measured[node_indices]
            if len(node_pred) > 1:
                node_corr, node_p = pearsonr(node_pred, node_meas)
                node_mse = np.mean((node_pred - node_meas) ** 2)
                node_mae = np.mean(np.abs(node_pred - node_meas))
                node_metrics[node] = {
                    'correlation': node_corr,
                    'p_value': node_p,
                    'mse': node_mse,
                    'mae': node_mae,
                    'n_points': len(node_pred)
                }
        metrics = {
            'overall': {
                'correlation': correlation,
                'p_value': p_value,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r_squared': r_squared,
                'n_points': len(predicted)
            },
            'per_node': node_metrics
        }
    else:
        print("Warning: No data available for evaluation.")

    # Plotting
    plot_paths = {}
    if generate_plots and len(predicted) > 0:
        # Scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_nodes = list(set(simulation_results['measured_nodes']))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_nodes)))
        node_color_map = dict(zip(unique_nodes, colors))
        for node in unique_nodes:
            node_indices = [i for i, n in enumerate(simulation_results['measured_nodes']) if n == node]
            node_pred = predicted[node_indices]
            node_meas = measured[node_indices]
            ax.scatter(node_meas, node_pred, c=[node_color_map[node]], label=node, alpha=0.7, s=60)
        min_val = min(np.min(predicted), np.min(measured))
        max_val = max(np.max(predicted), np.max(measured))
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(measured, predicted)
        x_reg = np.linspace(min_val, max_val, 100)
        y_reg = slope * x_reg + intercept
        ax.plot(x_reg, y_reg, 'g-', linewidth=2, alpha=0.8, label='Regression line')
        residuals = predicted - (slope * measured + intercept)
        mse_residuals = np.mean(residuals**2)
        # confidence_interval = 1.96 * np.sqrt(mse_residuals)
        # ax.fill_between(x_reg, y_reg - confidence_interval, y_reg + confidence_interval, alpha=0.2, color='green', label='95% Confidence interval')
        for i, (meas, pred, exp_id) in enumerate(zip(measured, predicted, simulation_results['experiment_ids'])):
            ax.annotate(f'E{exp_id}', (meas, pred), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, alpha=0.7)
        ax.set_xlabel('Experimental Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        # title = f'Predicted vs Experimental (r={metrics["overall"]["correlation"]:.3f}, p={metrics["overall"]["p_value"]:.3e}, MSE={metrics["overall"]["mse"]:.6f})'
        title = f'Predicted vs Experimental (r={metrics["overall"]["correlation"]:.3f}, p={metrics["overall"]["p_value"]:.3e})'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if len(unique_nodes) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.legend()
        # ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(np.min(measured) - 0.05, np.max(measured) + 0.05)
        ax.set_ylim(np.min(predicted) - 0.05, np.max(predicted) + 0.05)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "prediction_vs_experimental.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plot_paths['prediction_vs_experimental'] = plot_path
        plt.close(fig)
        # Residual plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        residuals = predicted - measured
        ax1.scatter(predicted, residuals, alpha=0.7, s=60)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals (Predicted - Measured)')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True, alpha=0.3)
        ax2.hist(residuals, bins=min(20, len(residuals)//3), alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        ax2.grid(True, alpha=0.3)
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        ax2.text(0.02, 0.98, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}', transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        residual_path = os.path.join(output_dir, "residual_analysis.png")
        plt.savefig(residual_path, dpi=300, bbox_inches='tight')
        plot_paths['residual_analysis'] = residual_path
        plt.close(fig)

    # Report
    report_path = None
    if generate_report and metrics:
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("PBN EVALUATION REPORT")
        report_lines.append("="*60)
        report_lines.append("")
        overall = metrics['overall']
        report_lines.append("OVERALL PERFORMANCE:")
        report_lines.append("-" * 20)
        report_lines.append(f"Pearson Correlation: {overall['correlation']:.4f}")
        report_lines.append(f"P-value: {overall['p_value']:.6e}")
        report_lines.append(f"R-squared: {overall['r_squared']:.4f}")
        report_lines.append(f"Mean Squared Error: {overall['mse']:.6f}")
        report_lines.append(f"Root Mean Squared Error: {overall['rmse']:.6f}")
        report_lines.append(f"Mean Absolute Error: {overall['mae']:.6f}")
        report_lines.append(f"Number of data points: {overall['n_points']}")
        report_lines.append("")
        if metrics['per_node']:
            report_lines.append("PER-NODE PERFORMANCE:")
            report_lines.append("-" * 20)
            for node, m in metrics['per_node'].items():
                report_lines.append(f"{node}:")
                report_lines.append(f"  Correlation: {m['correlation']:.4f} (p = {m['p_value']:.4e})")
                report_lines.append(f"  MSE: {m['mse']:.6f}")
                report_lines.append(f"  MAE: {m['mae']:.6f}")
                report_lines.append(f"  Data points: {m['n_points']}")
                report_lines.append("")
        report_text = "\n".join(report_lines)
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"Evaluation report saved to {report_path}")

    # Export CSV
    csv_path = os.path.join(output_dir, "detailed_results.csv")
    data = {
        'Experiment_ID': simulation_results['experiment_ids'],
        'Node': simulation_results['measured_nodes'],
        'Predicted_Value': simulation_results['predicted_values'],
        'Measured_Value': simulation_results['measured_values'],
        'Residual': np.array(simulation_results['predicted_values']) - np.array(simulation_results['measured_values']),
        'Absolute_Error': np.abs(np.array(simulation_results['predicted_values']) - np.array(simulation_results['measured_values']))
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Results exported to {csv_path}")

    return {
        'metrics': metrics,
        'plot_paths': plot_paths,
        'report_path': report_path,
        'csv_path': csv_path,
        'simulation_results': simulation_results
    }
