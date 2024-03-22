# BMatrix 
`Created: 1/29/2024` \
`Updated last on 3/16/2024`

### Required Packages 
Python, Numpy, Pandas, Re, Itertools, BooleanNetwork

### Installation
To install the BMatrix code, download it and import within Python file as 'BMatrix'
***

## Function Descriptions
### get_equations(file)
__Input__: A .txt file with the equations formatted `GENE = ! ( INHIBITOR | INHIBITOR ) & ( ACTIVATOR | ACTIVIATOR )` \
__Functionality__: Reads the .txt file by line, if the format is incorrect, it returns an error. \
__Output__: `equations` which is a list of strings, where each string represents one line of the .txt file
> ` Example: ['FLT3 = FLT3',
 'AKT = FLT3',
 'CEBPA = ! FLT3',
 'DNMT3A = DNMT3A',
 'GSK3B = ! AKT']`
 
### get_gene_dict(equations)
__Input__: Requires `equations` \
__Functionality__: From equations, takes all the values on the left side of the = sign and creates a dictionary with those values starting from 0. \
__Output__: `gene_dict` a dictionary which includes all the genes (nodes) that are included in the simulation
> `Example: {'FLT3': 0,
 'AKT': 1,
 'CEBPA': 2,
 'DNMT3A': 3,
 'GSK3B': 4}`

### get_upstream_genes(equations)
__Input__: Requires `equations` \
__Functionality__: Using `equations`, it takes the right side of the equations, and removes all symbols such as `| () &` and leaves only the genes. \
__Output__: `upstream_genes` which is a list of strings, where every string represents the upstream genes of the gene (node) at the same line in `equations`
> `Example: ['FLT3',
 'FLT3',
 ' FLT3',
 'DNMT3A']`
 
### get_connectivity_matrix(equations,gene_dict,upstream_genes)
__Input__: Requires `equations`, `gene_dict`, and `upstream_genes` \
__Functionality__: Takes the strings from `upstream_genes` and iterates over every gene in an individual string. For every gene in the string, it takes the value of that gene from `gene_dict` and appends the value to a result list. After every string is complete, it pads up the result list with -1. \
__Output__: `connectivity_matrix` which is a np.array 
> `Example: array([[ 0, -1, -1, -1],[ 0, -1, -1, -1],[ 0, -1, -1, -1],[ 3, -1, -1, -1]])`

### get_truth_table(equations,upstream_genes, show_functions = None)
__Input__: Requires `equations` and `upstream_genes`. `show_functions` is a flag that gives the user the option to print out the functions that the code uses in eval() to create the truth table. It is automatically set to `False`.\
__Functionality__: Takes the right side of the `equations` list per line and evaluates the Boolean function using all possible combinations of gene values. It appends this output as a tuple to a list. After every line is evaluated, -1 is added to make all the tuples the same length. \
__Output__: `truth_table` which is an np.array
> `Example: [[ 0  1 -1 -1 -1 -1 -1 -1][ 0  1 -1 -1 -1 -1 -1 -1][ 1  0 -1 -1 -1 -1 -1 -1][ 0  1 -1 -1 -1 -1 -1 -1]]`

### get_mutation_dict(file)
__Input__: Requires `file` which is a .txt file that includes all the genes and their mutations (which is simplified in this case as TSG = 0 and oncogenes = 1). The .txt file must have the format of `GENE = MUTATION` or else an error will occur. \
__Functionality__: For every line in the .txt file, it splits by the = sign and whatever is on the left is a key (in this case a key) and whatever is on the right of the = sign is the value. \
__Output__: `mutation_dict` which is a dictionary that has genes as the keys and mutations as the value.
> `Example: {'FLT3': 1, 'DNMT3A': 0, 'NPM1': 1}`

_This function can also be used to make perturbed_dict, which is needed when perturbed genes are involved in the simulation_

### get_knocking_genes(profile, mutation_dict, connectivity_matrix, gene_dict,perturbed_genes=None, perturbed_dict=None)
__Input__: Requires `profile` which contains the patient's mutation profile as a string (ex. 'FLT3,NPM1,DNMT3A'), `connectivity_matrix`, `gene_dict`

_This code does not require `perturbed_genes`, `perturbed_dict`, and `mutation_dict`. For both `perturbed_dict` and `perturbed_genes`, they are only to be used when perturbed genes are considered during the simulation. In this case, (where the effects of perturbed genes is to be measured) one can decide whether to include `mutation_dict`. Depends on the results one wants to get. `perturbed_genes` is formatted the same way as `profile` and `perturbed_dict` is formatted the same way as `mutation_dict`_ 

__Functionality__: \
The code reads the `profile` and splits the profile up into it's seperate genes. For `profile` and/or `peturbed_genes` the code splits up the string and removes any repeat values. 

_Perturbed genes + Mutated genes_: \
If both perturbed and mutated genes are involved, the code reads the `profile` and splits the profile and removes duplicates (which is then renamed `mutation_profile`). For every gene in the mutation_profile, if there is no genes, it returns 'no_mutations' and does not change the connectivity matrix or the inital state. If there is a gene, the code accesses the genes value in from `gene_dict`, and knocks the array in the connectivity matrix out to -1, and sets the gene's inital value equal to the gene's value in `mutation_dict`. Then, after all the genes in the `mutation_profile` are considered, the code does the same except it uses the perturbed genes. The main difference is that the perturbed genes DO NOT depend on the `profile`. It knocks in/out the perturbed genes regardless of patient profiles. 

 
_Peturbed genes ONLY_:\
If only perturbed genes are involved, the code reads the `perturbed_genes` and splits the profile and removes duplicates. For every gene in the `perturbed_genes`, if there is no genes, it returns 'no perturbed genes' and does not change the connectivity matrix or the inital state. If there is a gene, the code accesses the genes value in from `gene_dict`, and knocks the array in the connectivity matrix out to -1, and sets the gene's inital value equal to the gene's value in `perturbed_dict`.

_Mutated genes ONLY_: \
If only mutated genes are involved, the code reads the `profile` and splits the profile and removes duplicates (renamed `mutation_profile` to prevent rewriting the variable). For every gene in the `mutation_profile`, if there is no genes, it returns 'no mutated genes' and does not change the connectivity matrix or the inital state. If there is a gene, the code accesses the genes value in from `gene_dict`, and knocks the array in the connectivity matrix out to -1, and sets the gene's inital value equal to the gene's value in `mutation_dict`.

_The perturbed genes take superiority over mutated genes. For example, if a GENE = 1 in a `mutation_dict` and the GENE is perturbed with the definition of the GENE being GENE = 0 in `perturbed_dict` for all patients, the value of the GENE will be set to 0._ 

__Output__: The output is `mutated_connectivity_matrix` which is the individual patient's connectivity matrix, and `x0` which is the individual patient's unique inital state. 
> `Example:`\
> `mutated_connectivity_matrix': [ 0 -1 -1][ -1 -1 -1][ 0 -1 -1][ 3 -1 -1][ 1 -1 -1][ 5 -1 -1]`\
> `x0: [0,1,1,1,1]`

## _The calculating equations (which all start with `cal_`) are used post-simulation and calculate the phenotype scores and the final network score__

### get_cal_upstream_genes(equations):
__Input__: Requires `equations` _(These equations are from a .txt file that has the calculating functions Ex. Apoptosis = GENE)_ \
__Functionality__: Using `equations`, it takes the right side of the equations, and removes all symbols such as `| () &` and leaves only the genes. (Retains any duplicates) \
__Output__: `cal_upstream_genes` a list of lists, where every individual list is for one phenotype \
> `Example: [['BCL2', 'TP53'], ['CEBPA', 'ETV6', 'MEIS1']]`

### get_cal_functions(equations):
__Input__: Requires `equations` \
__Functionality__: Takes the right side of the `equations` (called `cal_functions` in the code) and replaces the Boolean symbols with + or -. In this case `! = -`, `| = +`, and `& = +`. The code then removes any parantheses in the functions, cleans them up (removing any extra spaces that came from removing the parantheses) and then returns the cleaned `cal_functions`. \ 
__Output__: `cal_functions` which are functions that have genes being added/subtracted for the total phenotype score. \
> `Example: ['TP53 + (-BCL2) ', ' CEBPA + ETV6 + (-MEIS1) '}]`

## _get_calculating_scores is to be used post-simulation, which requires the BooleanNetwork code_

### get_calculating_scores(network_traj, cal_functions, cal_upstream_genes, gene_dict, cal_range=None, scores_dict=None)
__Input__: Requires `network_traj` (from the BooleanNetwork simulation), `cal_functions`, `cal_upstream_genes` and `gene_dict`. \

_This code doesn't require `cal_range` or `scores_dict`. `cal_range` specifies a specific range of `network_traj` values to be used in calculating the scores, it is automatically set to the last 100,000 steps. `scores_dict` is a dictionary, where the keys are the phenotype names + `Network` score. It is automatically set to have the keys "Apoptosis", "Differentiation", "Proliferation", and "Network"_ \

__Functionality__: The code iterates over every function in `cal_functions` and retrieves the function's associated `cal_upstream_genes`. For each function, it extracts the genes and evaluates the function for each row in the `cal_range`. The results are stored in the scores_dict under the respective keys. After evaluating scores for every function in `cal_functions`, it calculates the 'Network' scores based on the formula: Proliferation - (Differentiation + Apoptosis). After calculating the Network scores, it calculates the mean of the 'Network' scores called the final score. \
__Output__: The output is a `scores_dict` which is a dictionary that has the scores for every phenotype + network for all of `cal_range` and `final_score` which is the mean of all the values of `scores_dict['Network']`. 
> `Example:`\
> `scores_dict: {'Apoptosis': [0,0,...],'Differentiation': [1,0,...],'Proliferation':[0,1,...],'Network':[0,1,...]}` \
> `final_score: 4.45318`
