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
        Experiments,Stimuli,Inhibitors,Measured_nodes,Measured_values
        1,TGFa,TNFa,"NFkB,ERK,C8,Akt","0.7,0.88,0,1"
        2,TNFa,TGFa,"NFkB,ERK,C8,Akt","0.3,0.12,1,0"
        3,"TGFa,TNFa",,"NFkB,ERK,C8,Akt","1,1,1,1"

        Note:
        - The measured values are normalized to be between 0 and 1 if not already.
        - Simply divide the measured values by the maximum value of the measured values.
        
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
                'inhibitors': list,
                'measured_nodes': list,
                'measured_values': list,
                'measurements': dict (node: value)
            }
        """
        df = pd.read_csv(csv_file)
        experiments = []
        
        for _, row in df.iterrows():
            experiment = {
                'id': row['Experiments'],
                'stimuli': ExperimentData._parse_node_list(row.get('Stimuli', '')),
                'inhibitors': ExperimentData._parse_node_list(row.get('Inhibitors', '')),
                'measured_nodes': ExperimentData._parse_node_list(row['Measured_nodes']),
                'measured_values': ExperimentData._parse_value_list(row['Measured_values'])
            }
            
            # Create measurements dictionary for easy access
            experiment['measurements'] = dict(zip(
                experiment['measured_nodes'], 
                experiment['measured_values']
            ))
            
            # Validate that we have the same number of nodes and values
            if len(experiment['measured_nodes']) != len(experiment['measured_values']):
                raise ValueError(f"Experiment {experiment['id']}: Number of measured nodes "
                               f"({len(experiment['measured_nodes'])}) does not match "
                               f"number of measured values ({len(experiment['measured_values'])})")
            
            # Normalize the measured values to be between 0 and 1 if not already
            if max(experiment['measured_values']) > 1:
                experiment['measured_values'] = [value / max(experiment['measured_values']) for value in experiment['measured_values']]
            else:
                experiment['measured_values'] = experiment['measured_values']
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
            If any experiment contains invalid node names
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
            invalid_measured = set(exp['measured_nodes']) - nodes
            if invalid_measured:
                raise ValueError(f"Experiment {exp['id']}: Measured nodes {invalid_measured} "
                               f"not found in the network")
            
            # Check value ranges
            for node, value in exp['measurements'].items():
                if not (0 <= value <= 1):
                    raise ValueError(f"Experiment {exp['id']}: Measured value {value} for node "
                                   f"{node} must be between 0 and 1")
        
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
            "value_ranges": {}
        }
        
        for exp in experiments:
            summary["unique_stimuli"].update(exp['stimuli'])
            summary["unique_inhibitors"].update(exp['inhibitors'])
            summary["unique_measured_nodes"].update(exp['measured_nodes'])
            
            for node, value in exp['measurements'].items():
                if node not in summary["value_ranges"]:
                    summary["value_ranges"][node] = {"min": value, "max": value}
                else:
                    summary["value_ranges"][node]["min"] = min(summary["value_ranges"][node]["min"], value)
                    summary["value_ranges"][node]["max"] = max(summary["value_ranges"][node]["max"], value)
        
        # Convert sets to lists for JSON serialization
        summary["unique_stimuli"] = list(summary["unique_stimuli"])
        summary["unique_inhibitors"] = list(summary["unique_inhibitors"])
        summary["unique_measured_nodes"] = list(summary["unique_measured_nodes"])
        
        return summary


if __name__ == "__main__":
    # Test with existing Trairatphisan data if available
    try:
        experiments = ExperimentData.load_from_csv('../../data/Trairatphisan2014_case3.csv')
        print("Successfully loaded Trairatphisan2014_case3.csv")
        # print(experiments)
        summary = ExperimentData.get_experiment_summary(experiments)
        print("Summary:", summary)
    except:
        print("Trairatphisan2014_case3.csv not found")
