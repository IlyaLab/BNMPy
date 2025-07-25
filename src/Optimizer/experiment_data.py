import pandas as pd
import numpy as np

class ExperimentData:
    """
    Handle experimental data from CSV files
    """
    
    @staticmethod
    def load_from_csv(csv_file):
        """
        Load experiments from CSV file
        
        CSV Format:
        Experiments,Stimuli,Stimuli_efficacy,Inhibitors,Inhibitors_efficacy,Measured_nodes,Measured_values
        1,TGFa,1,TNFa,1,"NFkB,ERK,C8,Akt","0.7,0.88,0,1"
        2,TNFa,1,TGFa,1,"NFkB,ERK,C8,Akt","0.3,0.12,1,0"
        3,"TGFa,TNFa","1,1",,,"NFkB,ERK,C8,Akt","1,1,1,1"
        4,"TGFa,TNFa","1,1",PI3K,0.7,"NFkB,ERK,C8,Akt","0.3,0.12,1,0"

        Note:
        - The measured values are normalized to be between 0 and 1 if not already.
        - Simply divide the measured values by the maximum value of the measured values.
        - Stimuli_efficacy and Inhibitors_efficacy are optional columns.
        - If efficacy is not specified, defaults to 1.0 (full efficacy).
        - Efficacy < 1 means the probability of achieving the target state is reduced.
        
        Parameters:
        -----------
        csv_file : str
            Path to CSV file with experimental data
            
        Returns:
        --------
        list
            List of experiment dictionaries with parsed data
            {
                'id': str,
                'stimuli': list,
                'stimuli_efficacy': list,
                'inhibitors': list,
                'inhibitors_efficacy': list,
                'measurements': dict (node: value)
            }
        """
        df = pd.read_csv(csv_file)
        experiments = []
        
        for _, row in df.iterrows():
            # Create measurements dictionary for easy access
            measured_nodes = ExperimentData._parse_node_list(row['Measured_nodes'])
            measured_values = ExperimentData._parse_value_list(row['Measured_values'])

            # Validate that we have the same number of nodes and values
            if len(measured_nodes) != len(measured_values):
                raise ValueError(f"Experiment {row['Experiments']}: Number of measured nodes "
                               f"({len(measured_nodes)}) does not match "
                               f"number of measured values ({len(measured_values)})")

            # Normalize values if necessary
            if measured_values and max(measured_values) > 1:
                max_val = max(measured_values)
                measured_values = [v / max_val for v in measured_values]

            # Parse stimuli and their efficacies
            stimuli = ExperimentData._parse_node_list(row.get('Stimuli', ''))
            stimuli_efficacy = ExperimentData._parse_value_list(row.get('Stimuli_efficacy', ''))
            
            # If no efficacy specified, default to 1.0 for all stimuli
            if not stimuli_efficacy and stimuli:
                stimuli_efficacy = [1.0] * len(stimuli)
            
            # Parse inhibitors and their efficacies
            inhibitors = ExperimentData._parse_node_list(row.get('Inhibitors', ''))
            inhibitors_efficacy = ExperimentData._parse_value_list(row.get('Inhibitors_efficacy', ''))
            
            # If no efficacy specified, default to 1.0 for all inhibitors
            if not inhibitors_efficacy and inhibitors:
                inhibitors_efficacy = [1.0] * len(inhibitors)

            experiment = {
                'id': row['Experiments'],
                'stimuli': stimuli,
                'stimuli_efficacy': stimuli_efficacy,
                'inhibitors': inhibitors,
                'inhibitors_efficacy': inhibitors_efficacy,
                'measurements': dict(zip(measured_nodes, measured_values))
            }
            
            experiments.append(experiment)
        
        return experiments
    
    @staticmethod
    def _parse_node_list(node_string):
        """
        Parse comma-separated node names from string
        Handle quoted strings and empty values
        
        Parameters:
        -----------
        node_string : str
            Comma-separated node names, possibly quoted
            
        Returns:
        --------
        list
            List of cleaned node names
        """
        if pd.isna(node_string) or node_string == '':
            return []
        
        # Remove quotes if present
        node_string = str(node_string).strip('"\'')
        
        # Split by comma and clean up
        nodes = [node.strip() for node in node_string.split(',') if node.strip()]
        return nodes
    
    @staticmethod
    def _parse_value_list(value_string):
        """
        Parse comma-separated numerical values from string
        
        Parameters:
        -----------
        value_string : str
            Comma-separated numerical values, possibly quoted
            
        Returns:
        --------
        list
            List of float values
        """
        if pd.isna(value_string) or value_string == '':
            return []
        
        # Remove quotes if present
        value_string = str(value_string).strip('"\'')
        
        # Split by comma and convert to float
        values = [float(val.strip()) for val in value_string.split(',') if val.strip()]
        return values
    
    @staticmethod
    def validate_experiments(experiments, node_dict):
        """
        Validate experiments against PBN structure
        
        Parameters:
        -----------
        experiments : list
            List of experiment dictionaries
        node_dict : dict
            Node name to index mapping
            
        Returns:
        --------
        bool
            True if all experiments are valid
            
        Raises:
        -------
        ValueError
            If any experiment contains invalid node names or efficacy values
        """
        nodes = set(node_dict.keys())
        
        for exp in experiments:
            # Check stimuli nodes
            invalid_stimuli = set(exp['stimuli']) - nodes
            if invalid_stimuli:
                raise ValueError(f"Experiment {exp['id']}: Stimuli nodes {invalid_stimuli} "
                               f"not found in the network")
            
            # Check inhibitor nodes
            invalid_inhibitors = set(exp['inhibitors']) - nodes
            if invalid_inhibitors:
                raise ValueError(f"Experiment {exp['id']}: Inhibitor nodes {invalid_inhibitors} "
                               f"not found in the network")
            
            # Check measured nodes
            invalid_measured = set(exp['measurements'].keys()) - nodes
            if invalid_measured:
                raise ValueError(f"Experiment {exp['id']}: Measured nodes {invalid_measured} "
                               f"not found in the network")
            
            # Check value ranges for measurements
            for node, value in exp['measurements'].items():
                if not (0 <= value <= 1):
                    raise ValueError(f"Experiment {exp['id']}: Measured value {value} for node "
                                   f"{node} must be between 0 and 1")
            
            # Check stimuli efficacy values
            if len(exp['stimuli_efficacy']) != len(exp['stimuli']):
                raise ValueError(f"Experiment {exp['id']}: Number of stimuli efficacy values "
                               f"({len(exp['stimuli_efficacy'])}) does not match "
                               f"number of stimuli ({len(exp['stimuli'])})")
            
            for i, efficacy in enumerate(exp['stimuli_efficacy']):
                if not (0 <= efficacy <= 1):
                    raise ValueError(f"Experiment {exp['id']}: Stimuli efficacy {efficacy} for "
                                   f"node {exp['stimuli'][i]} must be between 0 and 1")
            
            # Check inhibitor efficacy values
            if len(exp['inhibitors_efficacy']) != len(exp['inhibitors']):
                raise ValueError(f"Experiment {exp['id']}: Number of inhibitor efficacy values "
                               f"({len(exp['inhibitors_efficacy'])}) does not match "
                               f"number of inhibitors ({len(exp['inhibitors'])})")
            
            for i, efficacy in enumerate(exp['inhibitors_efficacy']):
                if not (0 <= efficacy <= 1):
                    raise ValueError(f"Experiment {exp['id']}: Inhibitor efficacy {efficacy} for "
                                   f"node {exp['inhibitors'][i]} must be between 0 and 1")
        
        return True
    

    @staticmethod
    def get_experiment_summary(experiments):
        """
        Get summary of experiments for debugging/inspection
        
        Parameters:
        -----------
        experiments : list
            List of experiment dictionaries
            
        Returns:
        --------
        dict
            Summary statistics about the experiments
        """
        if not experiments:
            return {"num_experiments": 0}
        
        summary = {
            "num_experiments": len(experiments),
            "unique_stimuli": set(),
            "unique_inhibitors": set(),
            "unique_measured_nodes": set(),
            "value_ranges": {},
            "stimuli_efficacy_ranges": {},
            "inhibitors_efficacy_ranges": {}
        }
        
        for exp in experiments:
            summary["unique_stimuli"].update(exp['stimuli'])
            summary["unique_inhibitors"].update(exp['inhibitors'])
            summary["unique_measured_nodes"].update(exp['measurements'].keys())
            
            for node, value in exp['measurements'].items():
                if node not in summary["value_ranges"]:
                    summary["value_ranges"][node] = {"min": value, "max": value}
                else:
                    summary["value_ranges"][node]["min"] = min(summary["value_ranges"][node]["min"], value)
                    summary["value_ranges"][node]["max"] = max(summary["value_ranges"][node]["max"], value)
            
            # Track efficacy ranges for stimuli
            for i, node in enumerate(exp['stimuli']):
                efficacy = exp['stimuli_efficacy'][i]
                if node not in summary["stimuli_efficacy_ranges"]:
                    summary["stimuli_efficacy_ranges"][node] = {"min": efficacy, "max": efficacy}
                else:
                    summary["stimuli_efficacy_ranges"][node]["min"] = min(summary["stimuli_efficacy_ranges"][node]["min"], efficacy)
                    summary["stimuli_efficacy_ranges"][node]["max"] = max(summary["stimuli_efficacy_ranges"][node]["max"], efficacy)
            
            # Track efficacy ranges for inhibitors
            for i, node in enumerate(exp['inhibitors']):
                efficacy = exp['inhibitors_efficacy'][i]
                if node not in summary["inhibitors_efficacy_ranges"]:
                    summary["inhibitors_efficacy_ranges"][node] = {"min": efficacy, "max": efficacy}
                else:
                    summary["inhibitors_efficacy_ranges"][node]["min"] = min(summary["inhibitors_efficacy_ranges"][node]["min"], efficacy)
                    summary["inhibitors_efficacy_ranges"][node]["max"] = max(summary["inhibitors_efficacy_ranges"][node]["max"], efficacy)
        
        # Convert sets to lists for JSON serialization
        summary["unique_stimuli"] = list(summary["unique_stimuli"])
        summary["unique_inhibitors"] = list(summary["unique_inhibitors"])
        summary["unique_measured_nodes"] = list(summary["unique_measured_nodes"])
        
        return summary

def extract_experiment_nodes(csv_file):
    """
    Extract measured and perturbed nodes from experimental CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with experimental data
        
    Returns:
    --------
    Tuple[Set[str], Set[str]]
        - measured_nodes: Set of node names that appear in Measured_nodes columns
        - perturbed_nodes: Set of node names that appear in Stimuli or Inhibitors columns
    """
    df = pd.read_csv(csv_file)
    measured_nodes = set()
    perturbed_nodes = set()
    
    for _, row in df.iterrows():
        # Extract measured nodes
        if 'Measured_nodes' in row and not pd.isna(row['Measured_nodes']):
            nodes = ExperimentData._parse_node_list(row['Measured_nodes'])
            measured_nodes.update(nodes)
        
        # Extract stimuli (perturbed nodes)
        if 'Stimuli' in row and not pd.isna(row['Stimuli']):
            nodes = ExperimentData._parse_node_list(row['Stimuli'])
            perturbed_nodes.update(nodes)
        
        # Extract inhibitors (perturbed nodes)
        if 'Inhibitors' in row and not pd.isna(row['Inhibitors']):
            nodes = ExperimentData._parse_node_list(row['Inhibitors'])
            perturbed_nodes.update(nodes)
    
    print(f"   Extracted {len(measured_nodes)} measured nodes: {measured_nodes}")
    print(f"   Extracted {len(perturbed_nodes)} perturbed nodes: {perturbed_nodes}")

    return measured_nodes, perturbed_nodes

def generate_experiments(pbn, experiment_csv: str, config: dict = None, output_csv: str = None, round_to: int = 4) -> pd.DataFrame:
    """
    Generate hypothesized experimental values using the current PBN parameters.
    
    This function simulates the experiments defined in the CSV file using the current
    PBN parameters and generates predicted values for the measured nodes.
    
    Parameters:
    -----------
    experiment_csv : str
        Path to the experiment CSV file
    output_csv : str, optional
        Path to save the generated results. If None, returns DataFrame without saving.
    round_to : int, optional
        Number of decimal places to round the predicted values to. Default is 4.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with original experiment data plus generated predicted values
    """
    # Load experiments
    experiments = ExperimentData.load_from_csv(experiment_csv)
    
    # Initialize steady state calculator
    from BNMPy.steady_state import SteadyStateCalculator
    steady_state_calc = SteadyStateCalculator(pbn)
    
    # Get steady state configuration
    default_config = {
    'steady_state': {
        'method': 'monte_carlo',
            'monte_carlo_params': {
                'n_runs': 3,
                'n_steps': 5000,
                'p_noise': 0.05
            }
    }
}
    ss_config = config.get('steady_state', default_config['steady_state'])
    results = []
    
    for i, experiment in enumerate(experiments):
        # Set experimental conditions
        steady_state_calc.set_experimental_conditions(
            stimuli=experiment['stimuli'],
            stimuli_efficacy=experiment['stimuli_efficacy'],
            inhibitors=experiment['inhibitors'],
            inhibitors_efficacy=experiment['inhibitors_efficacy']
        )
        
        # Calculate steady state
        method = ss_config.get('method', 'monte_carlo')
        if method == 'tsmc':
            params = ss_config.get('tsmc_params', {})
            steady_state = steady_state_calc.compute_stationary_tsmc(**params)
        else:  # monte_carlo
            params = ss_config.get('monte_carlo_params', {})
            steady_state = steady_state_calc.compute_stationary_mc(**params)
        
        # Reset experimental conditions
        steady_state_calc.reset_network_conditions()
        
        # Extract predicted values for measured nodes
        predicted_values = []
        for node in experiment['measurements'].keys():
            node_idx = pbn.nodeDict[node]
            predicted_values.append(round(steady_state[node_idx], round_to))
        
        # Create result row
        result_row = {
            'Experiments': experiment['id'],
            'Stimuli': ','.join(experiment['stimuli']) if experiment['stimuli'] else '',
            'Stimuli_efficacy': ','.join(map(str, experiment['stimuli_efficacy'])) if experiment['stimuli_efficacy'] else '',
            'Inhibitors': ','.join(experiment['inhibitors']) if experiment['inhibitors'] else '',
            'Inhibitors_efficacy': ','.join(map(str, experiment['inhibitors_efficacy'])) if experiment['inhibitors_efficacy'] else '',
            'Measured_nodes': ','.join(experiment['measurements'].keys()),
            'Measured_values': ','.join(map(str, experiment['measurements'].values())),
            'Predicted_values': ','.join(map(str, predicted_values))
        }
        
        results.append(result_row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save if output path provided
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Generated experiment results saved to: {output_csv}")
    
    return df