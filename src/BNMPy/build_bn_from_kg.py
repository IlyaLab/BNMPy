# building a boolean network

# TODO: given a KG subset, can we construct a boolean network?
from kgfe import graph_info, explanations, gene_names

# try using signor
def load_signor_network(gene_list, input_format="symbol", joiner='&'):
    """
    Creates a boolean network from SigNOR using all of the provided genes. Tries to build a connected Steiner subgraph...

    Args:
        gene_list - list of gene symbols, gene ids, or uniprot ids.
        input_format - "symbol", "id", or "uniprot"
        joiner - "&", "|", or "inhibitor_wins"
    """
    if ' ' not in joiner and (joiner == '&' or joiner == '|'):
        joiner = ' ' + joiner + ' '
    input_format = input_format.lower()
    filename = 'SIGNOR_formated.tsv'
    graph_table = graph_info.load_graph(filename)
    graph = graph_info.df_to_graph(graph_table, False)
    digraph = graph_info.df_to_graph(graph_table, True)
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
    tree = explanations.steiner_tree(graph, new_uniprot_list)
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
            inhibitor_string = ' & '.join(f'!{x}' for x in inhibitors)
            upregulator_string = ' | '.join(f'{x}' for x in upregulators)
            if inhibitor_string and upregulator_string:
                input_nodes_string = f'({inhibitor_string}) & ({upregulator_string})'
            elif inhibitor_string:
                input_nodes_string = f'{inhibitor_string}'
            elif upregulator_string:
                input_nodes_string = f'{upregulator_string}'
        else:
            input_nodes_string = joiner.join(input_nodes)
        if len(input_nodes) == 0:
            input_nodes_string = gene_name
        output_string = f'{gene_name} = {input_nodes_string}'
        bn_lines.append(output_string)
    return '\n'.join(bn_lines), all_relations


def merge_network(bn, new_equations, outer_joiner='&'):
    """
    Merges a BooleanNetwork object with a string or list of equations.
    """
    if isinstance(new_equations, str):
        new_equations = new_equations.split('\n')
    equations = bn.equations
    old_eq_dict = {}
    for eq in equations:
        terms = eq.split('=')
        key = terms[0].strip()
        old_eq_dict[key] = terms[1]
    new_eq_dict = {}
    for eq in new_equations:
        terms = eq.split('=')
        key = terms[0].strip()
        new_eq_dict[key] = terms[1]
    # just combine it in a primitive way...
    combined_keys = set(list(old_eq_dict.keys()) + list(new_eq_dict.keys()))
    combined_eqs = {}
    combined_eqs_list = []
    for key in combined_keys:
        new_term = ''
        if key in old_eq_dict and key not in new_eq_dict:
            new_term = old_eq_dict[key]
        elif key in new_eq_dict and key not in old_eq_dict:
            new_term = new_eq_dict[key]
        else:
            new_term = f'({old_eq_dict[key]}) {outer_joiner} ({new_eq_dict[key]})'
        combined_eqs[key] = new_term
        combined_eqs_list.append(f'{key} = {new_term}')
    return '\n'.join(combined_eqs_list)


def add_genes_to_network(bn, gene_list, input_format='symbol',
        joiner='inhibitor_wins', outer_joiner='&'):
    """
    Creates an augmented boolean network from SigNOR using all of the genes along with the genes already present in bn. Tries to build a connected Steiner subgraph...

    Args:
        bn - a BooleanNetwork object
        gene_list - list of gene symbols, gene ids, or uniprot ids.
        input_format - "symbol", "id", or "uniprot"
        joiner - "&", "|", or "inhibitor_wins" - used for the knowledge graph-derived network.
        outer_joiner - "&" or "|"
    """
    if ' ' not in joiner:
        joiner = ' ' + joiner + ' '
    # TODO: combine equations...
    new_equations, relations = load_signor_network(gene_list, input_format, joiner) 
    new_equations = new_equations.split('\n')
    return merge_network(bn, new_equations, outer_joiner)

