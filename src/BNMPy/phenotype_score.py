import pandas as pd
import numpy as np
import os
from typing import Union, Dict, List
base_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'KG_files')
file_name = 'significant_paths_to_phenotypes.txt'
file_path = os.path.join(folder_path, file_name)

def get_phenotypes(file_path = file_path):
    df = pd.read_csv(file_path, sep='\t')
    print(f'There are {df["EndNode"].nunique()} phenotypes')
    print(f'There are {df["QueryNode"].nunique()} genes')
    print(f'Available phenotypes: {df["EndNode"].unique()}')

def proxpath(genes, phenotypes = ['APOPTOSIS', 'DIFFERENTIATION', 'PROLIFERATION'], file_path = file_path):
    """
    This function is used to get the phenotype score for a given list of genes and phenotypes using ProxPath.
    Args:
        genes: list of genes
        phenotypes: list of phenotypes
        file_path: path to the ProxPath file
    Returns:
        pheno_df: dataframe with the phenotype score
    """
    # Load the ProxPath file
    df = pd.read_csv(file_path, sep='\t')
    
    # Function to find the closest gene to the EndNode
    def closest_gene(path_string, genes):
        # Split the path into components and reverse it (to start from the EndNode)
        components = path_string.split('--')[::-1]
        for component in components:
            # Check if the component contains any of the genes
            for gene in genes:
                if gene in component:
                    return gene
        return None
    
    # Filter rows for the given genes and phenotypes
    filtered_df = df[df['QueryNode'].isin(genes) & df['EndNode'].isin(phenotypes)].copy()
    
    # find the closest gene to the phenotype
    filtered_df['Closest_Gene'] = filtered_df.apply(lambda row: closest_gene(row['Path_String'], genes), axis=1)
    
    # Filter rows where QueryNode is the closest gene to the phenotype
    pheno_df = filtered_df[filtered_df['QueryNode'] == filtered_df['Closest_Gene']]
    pheno_df = pheno_df.drop(columns=['Closest_Gene'])

    # sort
    pheno_df = pheno_df.sort_values(by=['EndNode', 'QueryNode'])

    # remove rows where Final_Effect is 0
    pheno_df = pheno_df[pheno_df['Final_Effect'] != 0]

    # there may be different Final_Effect values for the same QueryNode and EndNode
    for gene in pheno_df['QueryNode'].unique():
        for phenotype in pheno_df['EndNode'].unique():
            if pheno_df[(pheno_df['QueryNode'] == gene) & (pheno_df['EndNode'] == phenotype)]['Final_Effect'].nunique() > 1:
                print(f"{gene} has dual effects on {phenotype}")

    # Keep only the rows with the lowest Path_Score for each gene-phenotype pair
    pheno_df = pheno_df.loc[pheno_df.groupby(['QueryNode', 'EndNode'])['Path_Score'].idxmin()]

    print(f'Path found for {pheno_df["EndNode"].nunique()} phenotypes: {pheno_df["EndNode"].unique()}')

    # save the filtered data to a new file
    # pheno_df.to_csv('Phenotypes.txt', sep='\t', index=False)

    return pheno_df

def _convert_simulation_to_dataframe(simulation_results, genes=None) -> pd.DataFrame:
    """
    Convert simulation results to a pandas DataFrame with gene names as columns.
    Each row represents a different state/condition.
    
    Parameters:
    -----------
    simulation_results : multiple formats
        - pd.Series or pd.DataFrame: converted to DataFrame format
        - np.ndarray: array of node values (requires genes for mapping)
        - dict: steady state results from BN with 'fixed_points' and 'cyclic_attractors'
    genes : list of str, optional
        List of gene names in the same order as they appear in simulation_results
        
    Returns:
    --------
    pd.DataFrame : DataFrame with states as rows and gene names as columns
                   Index contains state labels (e.g., 'Fixed_Point_1', 'Cycle_1_State_2')
    """
    # Normalize genes input (accept dict_keys, sets, etc.)
    if genes is not None and not isinstance(genes, list):
        genes = list(genes)

    # If already a pandas DataFrame, set columns if not present and genes provided
    if isinstance(simulation_results, pd.DataFrame):
        df = simulation_results.copy()
        if genes is not None:
            if df.columns is None or any(str(col).startswith("Node_") for col in df.columns):
                df.columns = genes
        return df

    # If pandas Series, use index as gene names if not provided
    if isinstance(simulation_results, pd.Series):
        if genes is None:
            return pd.DataFrame([simulation_results.values], columns=simulation_results.index, index=['State_1'])
        else:
            return pd.DataFrame([simulation_results.values], columns=genes, index=['State_1'])

    # Handle dictionary (BN steady state results with multiple attractors)
    if isinstance(simulation_results, dict):
        states_list = []
        state_labels = []
        
        # Add all fixed points
        fixed_points = simulation_results['fixed_points']
        for i, fp in enumerate(fixed_points):
            states_list.append(fp)
            state_labels.append(f'Fixed_Point_{i+1}')
        
        # Add all cyclic attractors (each state in each cycle)
        if 'cyclic_attractors' in simulation_results:
            cyclic_attractors = simulation_results['cyclic_attractors']
            for cycle_idx, cycle in enumerate(cyclic_attractors):
                for state_idx, state in enumerate(cycle):
                    states_list.append(state)
                    state_labels.append(f'Cycle_{cycle_idx+1}_State_{state_idx+1}')
        
        if len(states_list) == 0:
            raise ValueError("No attractors found in steady state results")
        
        # Convert to DataFrame
        if genes is None or len(genes) != len(states_list[0]):
            print("Genes not provided or do not match the number of nodes in the steady state results")
            columns = [f"Node_{i}" for i in range(len(states_list[0]))]
        else:
            columns = genes
        df = pd.DataFrame(states_list, columns=columns, index=state_labels)
        return df

    # Handle numpy array
    elif isinstance(simulation_results, np.ndarray):
        # Check if 1D or 2D
        if simulation_results.ndim == 1:
            # Single state
            if genes is None or len(genes) != len(simulation_results):
                columns = [f"Node_{i}" for i in range(len(simulation_results))]
            else:
                columns = genes
            return pd.DataFrame([simulation_results], columns=columns, index=['State_1'])
        elif simulation_results.ndim == 2:
            # Multiple states (rows are states, columns are nodes)
            if genes is None or len(genes) != simulation_results.shape[1]:
                columns = [f"Node_{i}" for i in range(simulation_results.shape[1])]
            else:
                columns = genes
            state_labels = [f'State_{i+1}' for i in range(simulation_results.shape[0])]
            return pd.DataFrame(simulation_results, columns=columns, index=state_labels)
        else:
            raise ValueError(f"Numpy array must be 1D or 2D, got {simulation_results.ndim}D")
    
    else:
        raise TypeError(f"Unsupported simulation_results type: {type(simulation_results)}")

def phenotype_scores(phenotypes = ['APOPTOSIS', 'DIFFERENTIATION', 'PROLIFERATION'], 
                    file_path = file_path, genes=None, simulation_results=None, network=None):
    """
    Calculate phenotype scores for given genes and phenotypes using ProxPath.
    If simulation_results is not provided, it will return the formula to calculate the score.
    Otherwise, it will return the phenotype score directly as a DataFrame.

    Parameters:
    -----------
    phenotypes : list of str, default=['APOPTOSIS', 'DIFFERENTIATION', 'PROLIFERATION']
        List of phenotype names
    file_path : str
        Path to the ProxPath file
    genes : list of str, optional
        List of gene names for which to calculate phenotype scores
    simulation_results : multiple formats, optional
        Simulation results in one of the following formats:
        - pd.Series or pd.DataFrame: with gene names as index/columns
        - np.ndarray: array of node values (order same as genes or network.nodeDict)
        - dict: steady state results from BN with 'fixed_points' and 'cyclic_attractors'
    network : optional
        Network object; only used to infer gene order if 'genes' is not provided.

    Returns:
    --------
    pd.DataFrame or dict
        If simulation_results is provided: DataFrame with states as rows and phenotypes as columns
        If simulation_results is None: dict with phenotype names as keys and formulas as values
        
    Examples:
    ---------
    >>> # With numpy array and genes parameter
    >>> steady_state = np.array([0.5, 0.8, 0.2])
    >>> scores = phenotype_scores(
    ...     genes=['TP53', 'MYC', 'BCL2'], 
    ...     simulation_results=steady_state
    ... )
    
    >>> # With network object (extracts genes automatically)
    >>> scores = phenotype_scores(simulation_results=steady_state, network=bn)
    
    >>> # With pandas Series (gene names already in index)
    >>> sim_results = pd.Series([0.5, 0.8], index=['TP53', 'MYC'])
    >>> scores = phenotype_scores(simulation_results=sim_results)
    """
    # Normalize genes to list
    if genes is None:
        # try to extract genes from network.nodeDict
        if network is not None and hasattr(network, 'nodeDict'):
            genes = list(network.nodeDict.keys())
        # or extract from simulation_results if it's a pandas DataFrame
        elif simulation_results is not None and hasattr(simulation_results, 'columns'):
            print(f"Extracted genes from simulation_results: {simulation_results.columns}")
            genes = list(simulation_results.columns)
        # or extract from simulation_results if it's a numpy array
        elif simulation_results is not None and hasattr(simulation_results, 'shape'):
            print(f"Extracted genes from simulation_results: {simulation_results.shape[1]}")
            genes = list(range(simulation_results.shape[1]))
        else:
            raise ValueError("'genes' is required to query ProxPath and order simulation_results.")
    if not isinstance(genes, list):
        genes = list(genes)
    # obtain the path from closest gene to the phenotypes
    pheno_df = proxpath(genes = genes, phenotypes = phenotypes, file_path = file_path)

    # Keep only the rows with the lowest Path_Score for each gene-phenotype pair
    pheno_unique = pheno_df.loc[pheno_df.groupby(['QueryNode', 'EndNode'])['Path_Score'].idxmin()]

    phenotypes = pheno_df['EndNode'].unique()
    
    # Convert simulation results to pandas DataFrame if provided
    if simulation_results is not None:
        sim_df = _convert_simulation_to_dataframe(simulation_results, genes=genes)
        
        # Initialize results dictionary for each state
        results = {state: {phenotype: 0.0 for phenotype in phenotypes} for state in sim_df.index}
    else:
        formulas_dict = {}
    
    # Loop through each phenotype
    for phenotype in phenotypes:
        
        # Filter the 'pheno' dataframe for the current phenotype
        filtered_pheno = pheno_unique[pheno_unique['EndNode'] == phenotype]

        terms = []
        # Loop through each row in the filtered 'pheno' dataframe
        for idx, row in filtered_pheno.iterrows():
            gene = row['QueryNode']
            effect = row['Final_Effect']
            
            if simulation_results is not None:
                # Check if the gene is in the simulation results
                if gene in sim_df.columns:
                    # Calculate score for each state
                    for state in sim_df.index:
                        results[state][phenotype] += sim_df.loc[state, gene] * effect
            else:
                # Build terms for the formula string, using appropriate sign placement
                if effect == 1:
                    terms.append(f"{gene}")
                elif effect == -1:
                    terms.append(f"- {gene}")
        
        if simulation_results is None:
            formula = ' + '.join(terms).replace('+ -', '- ')
            formulas_dict[phenotype] = formula

    # Return as DataFrame if simulation_results was provided
    if simulation_results is not None:
        # Convert results dict to DataFrame with states as rows and phenotypes as columns
        return pd.DataFrame.from_dict(results, orient='index')
    else:
        return formulas_dict