### Created by Tazein on 1/29/24
## Two sets of equations: the simulation equations (which are the genes in the network that get updated) 
## and the calculating equations (which would be proliferation, differentiation, apoptosis)

#pip install truth-table-generator
import numpy as np
import pandas as pd
import re
from itertools import product

#import the equations
#file = ''

################## for simulation equations ##################

def getting_equations(file):
    with open(file, 'r') as file:
        equations = file.readlines()

    #strips the file down to just the equations
    equations = [equation.strip() for equation in equations]
    return(equations)

def gene_dictionary(equations):
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

def getting_only_genes(equations): 
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
        translation_table = str.maketrans("", "", characters_to_remove)
        cleaned_expression = function.translate(translation_table)  
        values.append(cleaned_expression)
    only_genes = values
    
    return(only_genes)
    
def connectivity_matrix(equations,only_genes,gene_dict):    
    #now we actually make the connectivity matrix
    result_list = []

    for function in only_genes:
        genes = function.split()
        values = tuple([gene_dict[gene] for gene in genes])
        result_list.append(values)
    result_array = np.array(result_list, dtype=object) #they are not all the same length
    
    #now we fix the length by adding the -1 (aka the padding) 
    max_length = max(len(t) for t in result_array)
    varF = [tuple(np.pad(t, (0, max_length - len(t)), constant_values=-1)) for t in result_array]
    varF = np.array(varF)
    
    return(varF)

def extracting_truth_table(equations,only_genes):

    #get only the right side of the equations
    right_side = []
    for equation in equations:
        parts = equation.split('=')
        value = parts[1].strip()
        right_side.append(value)
    functions = right_side

    functions = [function.replace('!', 'not').replace('|', 'or').replace('&','and') for function in functions]

    truth = []
    var1 = []
    i = 1

    for i in range(len(equations)):
        function = functions[i]
        #print(function)
    
        variables = [only_genes[i]] #get the genes in the expression (ex: FLT3)
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

    #print(truth)

    truth_table = np.array(truth, dtype=object) #they are not all the same length
    #print(truth_table)

    #now we fix the length by adding the -1 (aka the padding) 
    max_length = max(len(t) for t in truth_table)
    truth_table = [tuple(np.pad(t, (0, max_length - len(t)), constant_values=-1)) for t in truth_table]
    truth_table = np.array(truth_table)
    
    return(truth_table)

################## for calculating equations ##################

#getting the equations uses the same function (getting_equations(file))
#makes calculating functions (which are used with eval() later) and only_genes for the functions


#assumes that cal_functions == len(scores_dict)
def calculating_scores(y, cal_functions, cal_only_genes, gene_dict, y_range=None, scores_dict=None):
    if scores_dict is None:
        scores_dict = {"Apoptosis": [], "Differentiation": [], "Proliferation": [], "Network": []}
        
    if y_range is None:
        y_range = y[-100000:]
    
    title = ["Apoptosis", "Differentiation", "Proliferation", "Network"]
    
    for i in range(len(cal_functions)):
        score_function = cal_functions[i]
        variables = cal_only_genes[i]
        scores = []  # List to store scores for this iteration

        for row in y_range:
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
    Apoptosis = scores_dict['Apoptosis']
    Differentiation = scores_dict['Differentiation']
    Proliferation = scores_dict['Proliferation']
    
    for i in range(len(y_range)): 
        output = Proliferation[i] - (Differentiation[i] + Apoptosis[i])
        scores.append(output)
        
    scores_dict['Network'] = scores
    final_score = np.mean(scores_dict['Network'])
    
    return (scores_dict,final_score)
  

def calculating_only_genes(equations):

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
        values.append(cleaned_expression)
    cal_only_genes = values
    
    for i in range(len(cal_only_genes)):
        cal_only_genes[i] = cal_only_genes[i].split()
    
    return cal_only_genes


def calculating_functions(equations):
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


################## knocking in/out genes ##################
def knocking_genes(profile, varF, gene_dict, mutations):
    ngenes = len(gene_dict)
    
    if mutations is None:
        mutations = {}
            
    mutation_varF = varF.copy()  # Create a copy of varF for each iteration    
    mutation_profile = list(set(profile.split(',')))  # Removes any repeat values 
    #print(mutation_profile)             
    
    x0 = np.random.randint(2, size=ngenes)  # Random initial state resets with every profile
        
    # Make the mutation_varF rows in mutation_profile all -1 
    for gene in mutation_profile:
        if len(gene) == 0:
            print('no_mutation')
        else:
            mutation_varF[[gene_dict[gene]], :] = -1  # Knock the varF to -1
            x0[gene_dict[gene]] = mutations.get(gene, 0)  # Setting that gene's value to mutation value
            
    return(mutation_varF,x0)
  
  #def calculating_functions(equations):
    
   # right_side = []
   # for equation in equations:
   #    parts = equation.split('=')
   #    value = parts[1].strip()
   #    right_side.append(value)
   # cal_functions = right_side

  #  cal_functions = [function.replace('!', 'not').replace('|', 'or').replace('&','and') for function in cal_functions]
    
  #  return cal_functions
