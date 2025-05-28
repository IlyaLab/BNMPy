import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import copy

class PBNTrainer:
    """
    Trainer for Probabilistic Boolean Networks (PBNs) used for drug response prediction.
    
    This class implements the algorithm for training PBNs to predict drug responses in 
    cell lines with various mutations.
    """
    
    def __init__(self, genes, candidate_functions, mutations, drugs, drug_targets, response_matrix, 
                 initial_state=None, lambda_sparsity=0.01, max_iterations=1000, convergence_threshold=1e-4):
        """
        Initialize the PBN trainer.
        
        Parameters:
        -----------
        genes : list
            List of gene names in the network G = {g_1, g_2, ..., g_n}
        candidate_functions : dict
            Dictionary mapping gene names to lists of candidate Boolean functions
            F_i = {f_{i1}, f_{i2}, ..., f_{im_i}}
        mutations : dict
            Dictionary mapping cell line IDs to lists of mutations (gene states)
        drugs : list
            List of drug names D = {d_1, d_2, ..., d_p}
        drug_targets : dict
            Dictionary mapping drug names to lists of target genes T(d_j) ⊂ G
        response_matrix : numpy.ndarray
            Matrix R where r_{ij} represents the response of cell line i to drug j
        lambda_sparsity : float
            Sparsity parameter λ to control model complexity
        max_iterations : int
            Maximum number of iterations for simulation
        convergence_threshold : float
            Threshold for convergence in optimization
        """
        self.genes = genes
        self.candidate_functions = candidate_functions
        self.mutations = mutations
        self.drugs = drugs
        self.drug_targets = drug_targets
        self.response_matrix = response_matrix
        self.lambda_sparsity = lambda_sparsity
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.initial_state = initial_state

        # Initialize probabilities uniformly
        self.probabilities = {}
        for gene, functions in candidate_functions.items():
            self.probabilities[gene] = np.zeros(len(functions))
    
    def configure_cell_line(self, cell_line_id):
        """
        Configure the network for a specific cell line based on its mutations.
        
        Parameters:
        -----------
        cell_line_id : str
            Identifier for the cell line
            
        Returns:
        --------
        dict
            Initial gene states for the cell line
        """
        # Use initial state if provided, otherwise default to 0
        gene_states = {gene: state for gene, state in self.initial_state.items()} if self.initial_state else {gene: 0 for gene in self.genes}
        
        for mutation in self.mutations.get(cell_line_id, []):
            gene = mutation['gene']
            if gene in self.genes:
                # Set gene state based on functional impact (activating: 1, inactivating: 0)
                gene_states[gene] = 1 if mutation['effect'] == 'activating' else 0
        
        return gene_states
        
    def simulate_drug_effect(self, cell_line_state, drug):
        """
        Simulate the effect of a drug on a cell line.
        
        Parameters:
        -----------
        cell_line_state : dict
            Current state of genes in the cell line
        drug : str
            Name of the drug
            
        Returns:
        --------
        float
            Predicted drug response (e.g., AUC/IC50)
        """
        state = dict(cell_line_state)
        
        # Set drug target genes to 0 (inhibition)
        for target in self.drug_targets.get(drug, []):
            if target in state:
                state[target] = 0
        
        # Run PBN simulation until steady state or max iterations
        for _ in range(self.max_iterations):
            new_state = self._update_state(state)
            if new_state == state:  # Steady state reached
                break
            state = new_state
        
        # Calculate activity score (here we use a simple sum of active genes)
        # In practice, this could be a more sophisticated pathway analysis
        activity_score = sum(state.values()) / len(state)
        
        # Convert activity score to predicted response (simple linear mapping for demonstration)
        # In practice, this mapping would be calibrated based on training data
        predicted_response = 1 - activity_score  # Assuming higher activity -> lower response
        
        return predicted_response
    
    def _update_state(self, current_state):
        """
        Update the network state according to the PBN rules.
        
        Parameters:
        -----------
        current_state : dict
            Current state of genes in the network
            
        Returns:
        --------
        dict
            Updated state after one step of simulation
        """
        new_state = dict(current_state)
        
        for gene in self.genes:
            if sum(self.probabilities[gene]) == 0:
                continue  # Skip genes with no active rules
                
            # Normalize probabilities
            gene_probs = self.probabilities[gene] / sum(self.probabilities[gene])
            
            # Choose a function based on probabilities
            function_idx = np.random.choice(len(self.candidate_functions[gene]), p=gene_probs)
            selected_function = self.candidate_functions[gene][function_idx]
            
            # Apply the selected function
            new_state[gene] = self._evaluate_function(selected_function, current_state)
        
        return new_state
    
    def _evaluate_function(self, function, state):
        """
        Evaluate a Boolean function given the current network state.
        
        Parameters:
        -----------
        function : callable
            Boolean function to evaluate
        state : dict
            Current state of genes in the network
            
        Returns:
        --------
        int
            Result of the function (0 or 1)
        """
        # This is a placeholder - the actual implementation would depend on
        # how Boolean functions are represented in your system
        return function(state)
    
    def optimize(self, n_folds=5):
        """
        Optimize the rule probabilities using k-fold cross-validation.
        
        Parameters:
        -----------
        n_folds : int
            Number of folds for cross-validation
            
        Returns:
        --------
        dict
            Optimized probabilities for each gene's rules
        """
        # Prepare cross-validation
        cell_line_ids = list(self.mutations.keys())
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        best_probabilities = None
        best_error = float('inf')
        
        for train_idx, test_idx in kf.split(cell_line_ids):
            # Get train and test cell lines
            train_cell_lines = [cell_line_ids[i] for i in train_idx]
            test_cell_lines = [cell_line_ids[i] for i in test_idx]
            
            # Reset probabilities
            current_probs = {gene: np.zeros(len(funcs)) for gene, funcs in self.candidate_functions.items()}
            
            # Greedy forward selection
            improvement = True
            while improvement:
                improvement = False
                best_improvement = 0
                best_gene = None
                best_func_idx = None
                
                # Try adding each possible rule
                for gene in self.genes:
                    for func_idx in range(len(self.candidate_functions[gene])):
                        # Temporarily add this rule
                        temp_probs = {g: p.copy() for g, p in current_probs.items()}
                        temp_probs[gene][func_idx] += 1
                        
                        # Calculate error with this rule added
                        self.probabilities = temp_probs
                        error = self._calculate_error(train_cell_lines)
                        
                        # Add sparsity penalty
                        non_zero_count = sum(np.count_nonzero(probs) for probs in temp_probs.values())
                        total_error = error + self.lambda_sparsity * non_zero_count
                        
                        # Calculate improvement
                        current_error = self._calculate_error(train_cell_lines)
                        current_non_zero = sum(np.count_nonzero(probs) for probs in current_probs.values())
                        current_total = current_error + self.lambda_sparsity * current_non_zero
                        
                        improvement_value = current_total - total_error
                        
                        if improvement_value > best_improvement:
                            best_improvement = improvement_value
                            best_gene = gene
                            best_func_idx = func_idx
                
                # If we found an improvement, apply it
                if best_improvement > self.convergence_threshold:
                    current_probs[best_gene][best_func_idx] += 1
                    improvement = True
            
            # Normalize the probabilities
            for gene in self.genes:
                if sum(current_probs[gene]) > 0:
                    current_probs[gene] = current_probs[gene] / sum(current_probs[gene])
            
            # Test the model on test set
            self.probabilities = current_probs
            test_error = self._calculate_error(test_cell_lines)
            
            if test_error < best_error:
                best_error = test_error
                best_probabilities = {gene: probs.copy() for gene, probs in current_probs.items()}
        
        self.probabilities = best_probabilities
        return best_probabilities
    
    def _calculate_error(self, cell_line_ids):
        """
        Calculate prediction error for a set of cell lines.
        
        Parameters:
        -----------
        cell_line_ids : list
            List of cell line IDs to evaluate
            
        Returns:
        --------
        float
            Mean squared error of predictions
        """
        true_responses = []
        predicted_responses = []
        
        for cl_idx, cell_line_id in enumerate(cell_line_ids):
            cell_line_state = self.configure_cell_line(cell_line_id)
            
            for drug_idx, drug in enumerate(self.drugs):
                # Get true response
                true_response = self.response_matrix[cl_idx, drug_idx]
                true_responses.append(true_response)
                
                # Get predicted response
                predicted_response = self.simulate_drug_effect(cell_line_state, drug)
                predicted_responses.append(predicted_response)
        
        return mean_squared_error(true_responses, predicted_responses)
    
    def predict(self, cell_line_id, drug):
        """
        Predict the response of a cell line to a drug.
        
        Parameters:
        -----------
        cell_line_id : str
            Identifier for the cell line
        drug : str
            Name of the drug
            
        Returns:
        --------
        float
            Predicted drug response
        """
        cell_line_state = self.configure_cell_line(cell_line_id)
        return self.simulate_drug_effect(cell_line_state, drug)
