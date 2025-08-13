# building a boolean network

# given a KG subset, can we construct a boolean network?

# basic cache
GRAPH_TABLES = {}

def all_boolean_combos(n):
    "Generates all boolean strings of length n, as tuples of 1s and 0s."
    if n == 0:
        return []
    elif n == 1:
        return [(0,), (1,)]
    else:
        results = all_boolean_combos(n-1)
        new_results = []
        for r in results:
            new_results.append((0,) + r)
            new_results.append((1,) + r)
        return new_results


# try using signor
def load_signor_network(gene_list, input_format="symbol", joiner='&', kg_filename='SIGNOR_2025_08_12.tsv',
        only_proteins=True):
    """
    Creates a boolean network from SigNOR using all of the provided genes. Tries to build a connected Steiner subgraph...

    Args:
        gene_list - list of gene symbols, gene ids, or uniprot ids.
        input_format - "symbol", "id", or "uniprot"
        joiner - "&", "|", "inhibitor_wins", or "majority", or "plurality" (difference between the last two: in plurality, a tie indicates 1, in majority, a tie indicates 0)
        kg_filename - "SIGNOR_2025_08_12.tsv" by default. "SIGNOR_formatted.tsv" can also be used (this is an older version of SigNOR).
        only_proteins - whether to only use protein nodes in SIGNOR for getting the subgraph (default: True)
    """
    from . import graph_info, steiner_tree, gene_names
    if ' ' not in joiner and (joiner == '&' or joiner == '|'):
        joiner = ' ' + joiner + ' '
    input_format = input_format.lower()
    if kg_filename in GRAPH_TABLES:
        graph_table = GRAPH_TABLES[kg_filename]
    else:
        graph_table = graph_info.load_graph(kg_filename)
        GRAPH_TABLES[kg_filename] = graph_table
    graph = graph_info.df_to_graph(graph_table, False)
    digraph = graph_info.df_to_graph(graph_table, True)
    if only_proteins:
        protein_names = [n['name'] for n in graph_info.nodes_in_category(graph, 'protein')]
        graph = graph.induced_subgraph(protein_names)
        digraph = digraph.induced_subgraph(protein_names)
    # get a graph subset
    # signor names are uniprot, of the format UNIPROT::[uniprot ID]
    # feature_name is the gene name
    uniprot_list = []
    if input_format == 'symbol':
        id_list = gene_names.get_ids(gene_list)
        print(f"number of genes found: {len(id_list)}")
        print(id_list)
        uniprot_list = gene_names.gene_ids_to_uniprot(id_list)
    elif input_format == 'id' or input_format == 'gene_id':
        uniprot_list = gene_names.gene_ids_to_uniprot(gene_list)
    else:
        uniprot_list = gene_list
    uniprot_list = ['UNIPROT::'+x for x in uniprot_list]
    # print(uniprot_list)
    # filter uniprot list based on existence in the graph
    new_uniprot_list = []
    for u in uniprot_list:
        try:
            graph.vs.find(u)
            new_uniprot_list.append(u)
        except:
            continue
    # print(new_uniprot_list)
    # Steiner subgraph - get a tree using a connected thing...
    tree = steiner_tree.steiner_tree(graph, new_uniprot_list)
    subgraph = digraph.induced_subgraph([n['name'] for n in tree.vs])
    # use the subgraph to build a connected thing
    # TODO: figure out the rule for combining inputs
    bn_lines = []
    # inhibitors
    # get all input edges
    all_relations = []
    for n in subgraph.vs:
        inhibitors = []
        upregulators = []
        incoming_genes = set()
        gene_name = n.attributes()['feature_name']
        in_edges = subgraph.incident(n, mode='in')
        input_nodes = []
        for e in in_edges:
            e = subgraph.es[e]
            in_node = subgraph.vs[e.source].attributes()['feature_name']
            if in_node in incoming_genes:
                continue
            incoming_genes.add(in_node)
            predicate = e.attributes()['predicate']
            if 'down-regulates' in predicate:
                input_nodes.append(f'(! {in_node})')
                inhibitors.append(in_node)
                all_relations.append((in_node, gene_name, 'inhibit'))
            elif 'up-regulates' in predicate:
                input_nodes.append(f'({in_node})')
                upregulators.append(in_node)
                all_relations.append((in_node, gene_name, 'activate'))
        if joiner == 'inhibitor_wins':
            input_nodes_string = ''
            if len(inhibitors) > 1:
                inhibitor_string = ' & '.join(f'!{x}' for x in inhibitors)
                inhibitor_string =f'({inhibitor_string})'
            elif len(inhibitors) == 1:
                inhibitor_string = f'!{inhibitors[0]}'
            else:
                inhibitor_string = ''

            if len(upregulators) > 1:
                upregulator_string = ' | '.join(f'{x}' for x in upregulators)
                upregulator_string = f'({upregulator_string})'
            elif len(upregulators) == 1:
                upregulator_string = f'{upregulators[0]}'
            else:
                upregulator_string = ''

            if inhibitor_string and upregulator_string:
                input_nodes_string = f'{inhibitor_string} & {upregulator_string}'
            elif inhibitor_string:
                input_nodes_string = f'{inhibitor_string}'
            elif upregulator_string:
                input_nodes_string = f'{upregulator_string}'
            else:
                input_nodes_string = gene_name
        elif joiner == 'majority' or joiner == 'plurality':
            # a majority vote system for upregulators and downregulators
            # node is true if activators - repressors > 0
            input_nodes_string = ''
            up_down = upregulators + inhibitors
            all_outputs = []
            for permutation in all_boolean_combos(len(up_down)):
                upreg = sum(permutation[:len(upregulators)])
                downreg = sum(permutation[len(upregulators):])
                if (joiner == 'majority' and  upreg > downreg) or (joiner == 'plurality' and upreg >= downreg) :
                    gene_string = ' & '.join('!'+g if v==0 else g for v, g in zip(permutation, up_down))
                    all_outputs.append('('+gene_string+')')
            input_nodes_string = ' | '.join(all_outputs)
            if len(input_nodes_string) == 0:
                input_nodes_string = '0'
        else:
            input_nodes_string = joiner.join(input_nodes)

        if len(input_nodes) == 0:
            input_nodes_string = gene_name
        output_string = f'{gene_name} = {input_nodes_string}'
        bn_lines.append(output_string)
    # order the equations by key alphabetically
    bn_lines.sort(key=lambda x: x.split('=')[0])
    return '\n'.join(bn_lines), all_relations


def merge_PBN_string(original_string, KG_string, prob=0.5):
    """
    Merge the original model and the KG model to a PBN
    prob: probability of the equations from the original model
    """
    # Parse equations from both models
    original_equations = {}
    for line in original_string.strip().split('\n'):
        if '=' in line:
            target, rule = line.split('=', 1)
            original_equations[target.strip()] = rule.strip()
    
    kg_equations = {}
    for line in KG_string.strip().split('\n'):
        if '=' in line:
            target, rule = line.split('=', 1)
            kg_equations[target.strip()] = rule.strip()
    
    # Merge equations
    merged_equations = []
    all_targets = set(original_equations.keys()) | set(kg_equations.keys())
    
    for target in all_targets:
        if target in original_equations and target in kg_equations:
            if original_equations[target] == kg_equations[target]:
                # Both models have the same equation for this target
                merged_equations.append(f"{target} = {original_equations[target]}, 1")
            else:
                # Use both with specified probabilities
                merged_equations.append(f"{target} = {original_equations[target]}, {prob}")
                merged_equations.append(f"{target} = {kg_equations[target]}, {1-prob}")
        elif target in original_equations:
            # Only original model has this target
            merged_equations.append(f"{target} = {original_equations[target]}, 1")
            print(f"Only original model has this target: {target}")
        else:
            # Only KG model has this target
            print("Only KG model has this target:", target)
            merged_equations.append(f"{target} = {kg_equations[target]}, 1")
    
    # remove equation with prob = 0
    merged_equations = [eq for eq in merged_equations if eq.split(',')[1] != ' 0']
    merged_string = '\n'.join(merged_equations)
    return merged_string
