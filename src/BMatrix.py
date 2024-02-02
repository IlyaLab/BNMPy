### Created by Tazein on 1/29/24
## Goal is to create the truth tables from the DNMT3A subgroup equations

#pip install truth-table-generator
import numpy as np
import pandas as pd
import re
from itertools import product

#import the equations
file = 'C:/Users/15167/OneDrive/Documents/ISB/dnmt3a_equations.txt'

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

def only_function_genes(equations): 
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

def truth_table(equations,only_genes):

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
