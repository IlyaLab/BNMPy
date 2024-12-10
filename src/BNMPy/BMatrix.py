## Created by Tazein on 1/29/24
"""
Resources for loading boolean networks from files (strings)
"""

import numpy as np
import pandas as pd
import re
from itertools import product

################## equations for simulation variables ##################

BUILT_INS = {'0', '1', 'True', 'False'}

def get_equations(file):
    with open(file, 'r') as file:
        equations = file.readlines()
    equations = [equation.strip() for equation in equations if len(equation.strip()) > 0]
    return equations

def get_gene_dict(equations):
    left_side = []
    
    for equation in equations:
        parts = equation.split('=')
        value = parts[0].strip()
        if value not in left_side:
            left_side.append(value)

    genes = left_side
    
    # making a dictionary for the genes starting from 0
    gene_dict = {gene: i for i, gene in enumerate(genes)}
    return(gene_dict)

def get_upstream_genes(equations): 
    "Returns a string of gene names, space-separated, for each of the equations."
    #get only the right side of the equations
    right_side = []
    for equation in equations:
        parts = equation.split('=')
        value = parts[1].strip()
        right_side.append(value)
    functions = right_side

    #getting rid of the Boolean characters ! | & and ()
    characters_to_remove = "!|&()"
    values = []
    for function in functions:
        translation_table = str.maketrans({c: ' ' for c in characters_to_remove})
        cleaned_expression = function.translate(translation_table) 
        tokens = list(set(cleaned_expression.split()))
        tokens = [x for x in tokens if x not in BUILT_INS]
        cleaned_expression = ' '.join(tokens)
        values.append(cleaned_expression)
    upstream_genes = values

    return(upstream_genes)
    
def get_connectivity_matrix(equations,upstream_genes,gene_dict):    
    #now we actually make the connectivity matrix
    result_list = []

    for function in upstream_genes:
        genes = function.split()
        values = tuple([gene_dict[gene] for gene in genes])
        result_list.append(values)
    result_array = np.array(result_list, dtype=object) #they are not all the same length
    
    #now we fix the length by adding the -1 (aka the padding) 
    max_length = max(len(t) for t in result_array)
    connectivity_matrix = [tuple(np.pad(t, (0, max_length - len(t)), constant_values=-1)) for t in result_array]
    connectivity_matrix = np.array(connectivity_matrix, dtype=int)
    
    return(connectivity_matrix)

def get_truth_table(equations,upstream_genes,show_functions=None):
    
    if show_functions is None: 
        show_functions = False

    #get only the right side of the equations
    right_side = []
    for equation in equations:
        parts = equation.split('=')
        value = parts[1].strip()
        right_side.append(value)
    functions = right_side

    functions = [function.replace('!', ' not ').replace('|', ' or ').replace('&',' and ') for function in functions]
        
    truth = []
    var1 = []
    i = 1

    for i in range(len(equations)):
        function = functions[i]   
        if show_functions is False:
            pass
        else:
            print(function)
            
        variables = [upstream_genes[i]] #get the genes in the expression (ex: FLT3)
        variables = variables[0].split()
        combinations = product([0, 1], repeat=len(variables)) #gets all possiblities
        i += 1
    
        for combo in combinations:
            values = dict(zip(variables, combo))
        
            if len(variables) == 1 and variables[0] == function: #if the node is equal to itself (aka FLT3=FLT3)
                output = values[variables[0]]

            else:
                output = (int(eval(function, values))) #evaluates the equations

            var1.append(output) #adds the output to var1
        
        truth.append(tuple(var1))
        var1 = []

    truth_table = np.array(truth, dtype=object) #they are not all the same length

    #now we fix the length by adding the -1 (aka the padding) 
    max_length = max(len(t) for t in truth_table)
    truth_table = [tuple(np.pad(t, (0, max_length - len(t)), constant_values=-1)) for t in truth_table]
    truth_table = np.array(truth_table)
    
    return(truth_table)

################## knocking in/out genes after creating variables but before simulation ##################

##can also use mutation_dict for perturbed_dict, just replace the file 
def get_mutation_dict(file):
    mutation_dict = {}
    
    with open(file) as f:
        for line in f:
            if bool(re.search('=', line)) == False: #there is no = sign
                print('There is a formatting error: ' + str(line) + '\nMake sure that it is formatted with an equal sign. For example: FLT3 = 1')
                return
            
            key, val = line.split("=")
            mutation_dict[key.strip()] = int(val.strip())
    
    return(mutation_dict)

def get_knocking_genes(profile, mutation_dict, connectivity_matrix, gene_dict, perturbed_genes=None, perturbed_dict=None):
    ngenes = len(gene_dict)
    mutated_connectivity_matrix = connectivity_matrix.copy()  # Create a copy of connectivity_matrix for each iteration 
    x0 = np.random.randint(2, size=ngenes)  # Random initial state resets with every profile

    
    if perturbed_genes is None: 
        perturbed_genes = []
        
    if perturbed_dict is None:
        perturbed_dict = {}
        
    if profile is not None: #if there is a profile
        mutation_profile = list(set(profile.split(',')))  # Removes any repeat values 
        
    if profile is not None and mutation_dict is None: #there are no mutation_dict (aka no mutations) 
        mutation_profile = ''
            
    if perturbed_genes is not None: #if there are perturbed genes
        perturbed_genes = perturbed_genes if isinstance(perturbed_genes, list) else perturbed_genes.split(',')
        perturbed_genes = list(set(perturbed_genes))  # Removes any repeat values

        
    # Setting that gene's value to wild value
    for gene in mutation_dict:
        if gene == '' or gene == 'NA':
            print('no_mutation')
        else:
            if ( mutation_dict[gene] > 0  ) :
                x0[gene_dict[gene]] = 0 
            else :
                x0[gene_dict[gene]] = 1

    # Make the mutated_connectivity_matrix rows in mutation_profile all -1 
    for gene in mutation_profile:
        if gene == '' or gene == 'NA':
            print('no_mutation')
        else:
            mutated_connectivity_matrix[[gene_dict[gene]], :] = -1  # Knock the connectivity_matrix to -1
            x0[gene_dict[gene]] = mutation_dict.get(gene, 0)  # Setting that gene's value to mutation value
            
    for gene in perturbed_genes:
        if len(gene) == 0:
            print('no perturbed genes in the simulation')
        else:
            mutated_connectivity_matrix[[gene_dict[gene]], :] = -1  # Knock the connectivity_matrix to -1
            x0[gene_dict[gene]] = perturbed_dict.get(gene, 0)  # Setting that gene's value to mutation value
            
    return(mutated_connectivity_matrix,x0)


def load_network_from_file(filename, initial_state=None):
    """
    Given a file representing a boolean network, this generates a BooleanNetwork object.

    Formatting:

    - all genes must have their own equation (sometimes the equation is just A = A)
    - all 
    """
    from .booleanNetwork import BooleanNetwork
    equations = get_equations(filename)
    ngenes = len(equations)
    gene_dict = get_gene_dict(equations)
    upstream_genes = get_upstream_genes(equations)
    connectivity_matrix = get_connectivity_matrix(equations, upstream_genes, gene_dict)
    truth_table = get_truth_table(equations, upstream_genes)
    if initial_state is None:
        print('No initial state provided, using a random initial state')
        x0 = np.random.randint(2, size=ngenes) #random inital state 
    else:
        x0 = np.array(initial_state)
    network = BooleanNetwork( ngenes , connectivity_matrix, truth_table, x0,
            nodeDict=gene_dict)
    # create a Boolean network object
    return network

# TODO: implement constant values
def load_network_from_string(network_string, initial_state=None):
    """
    Given a file representing a boolean network, this generates a BooleanNetwork object.

    Formatting:

    - all genes must have their own equation (sometimes the equation is just A = A)
    - all 
    """
    from .booleanNetwork import BooleanNetwork
    equations = [x.strip() for x in network_string.strip().split('\n')]
    ngenes = len(equations)
    gene_dict = get_gene_dict(equations)
    upstream_genes = get_upstream_genes(equations)
    connectivity_matrix = get_connectivity_matrix(equations, upstream_genes, gene_dict)
    truth_table = get_truth_table(equations, upstream_genes)
    if initial_state is None:
        print('No initial state provided, using a random initial state')
        x0 = np.random.randint(2, size=ngenes) #random inital state 
    else:
        x0 = np.array(initial_state)
    network = BooleanNetwork( ngenes , connectivity_matrix, truth_table, x0,
            nodeDict=gene_dict)
    # create a Boolean network object
    return network

################## equations for calculating phenotype and network ##################

#getting the equations uses the same function (equations(file))

def get_cal_upstream_genes(equations):
    right_side = []
    for equation in equations:
        parts = equation.split('=')
        value = parts[1].strip()
        right_side.append(value)
    functions = right_side

    #getting rid of the Boolean characters ! | & and ()
    characters_to_remove = "!|&()"
    values = []
    for function in functions:
        translation_table = str.maketrans("", "", characters_to_remove)
        cleaned_expression = function.translate(translation_table)  
        cleaned_expression = ' '.join(list(set(cleaned_expression.split())))
        values.append(cleaned_expression)
    cal_upstream_genes = values
    
    for i in range(len(cal_upstream_genes)):
        cal_upstream_genes[i] = cal_upstream_genes[i].split()
    
    return cal_upstream_genes

def get_cal_functions(equations):
    right_side = []
    for equation in equations:
        parts = equation.split('=')
        value = parts[1].strip()
        right_side.append(value)
    cal_functions = right_side

    cal_functions = [function.replace('!', '-').replace('|', '+').replace('&','+') for function in cal_functions]

    characters_to_remove = "()"
    values = []
    for function in cal_functions:
        translation_table = str.maketrans("", "", characters_to_remove)
        cleaned_expression = function.translate(translation_table)  
        values.append(cleaned_expression)
        cal_functions = values
    
    #cleaning up the cal_functions format so it can be eval()
    cal_functions = [function.replace(' - ', ' -') for function in cal_functions]
    cal_functions = [function.replace('- ', '-') for function in cal_functions]
    cal_functions = [function.replace('- ', '-') for function in cal_functions]
    cal_functions = [function.replace('  ', ' ') for function in cal_functions]

    values = []
    for function in cal_functions:
        new_func = re.sub(r'(-\w+)', r'(\1)', function)
        values.append(new_func)
        cal_functions = values
        
    return(cal_functions)

#assumes that cal_functions == len(scores_dict)
def get_calculating_scores(network_traj, cal_functions, cal_upstream_genes, gene_dict, cal_range=None, scores_dict=None, title=None):
    if scores_dict is None:
        scores_dict = {"Apoptosis": [], "Differentiation": [], "Proliferation": [], "Network": []}
        
    if cal_range is None:
        cal_range = network_traj[-100000:]
    
    if title is None:
        title = ["Apoptosis", "Differentiation", "Proliferation", "Network"]
    
    for i in range(len(cal_functions)):
        score_function = cal_functions[i]
        variables = cal_upstream_genes[i]
        scores = []  # List to store scores for this iteration

        for row in cal_range:
            gene_values = []  # Clear gene_values for each row
            for gene in variables:
                value = row[gene_dict[gene]]
                gene_values.append(value)

            values = dict(zip(variables, gene_values))
            output = int(eval(score_function, values))

            scores.append(output)  # Append the score to the list for this iteration

        scores_dict[title[i]] = scores
        
    # Calculate the 'Network' scores
    scores = []
    Apoptosis = np.mean( scores_dict['Apoptosis'] )
    Differentiation = np.mean( scores_dict['Differentiation'] )
    Proliferation = np.mean( scores_dict['Proliferation'] )
    
    final_scores_dict = {} 
    
    final_scores_dict['Apoptosis']  = Apoptosis
    final_scores_dict['Differentiation'] = Differentiation
    final_scores_dict['Proliferation'] = Proliferation
    #for i in range(len(cal_range)): 
    #output = Proliferation[i] - (Differentiation[i] + Apoptosis[i])
    #scores.append(output)
    
    final_score = Proliferation - Differentiation - Apoptosis
    final_scores_dict['Network'] = final_score #scores
    #final_score = np.mean(scores_dict['Network'])
    
    return (final_scores_dict,final_score)
