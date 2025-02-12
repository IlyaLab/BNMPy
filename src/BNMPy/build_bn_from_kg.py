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
        joiner - "&" or "|"
    """
    if ' ' not in joiner:
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
        uniprot_list = gene_names.gene_ids_to_uniprot(id_list)
    elif input_format == 'id' or input_format == 'gene_id':
        uniprot_list = gene_names.gene_ids_to_uniprot(gene_list)
    else:
        uniprot_list = gene_list
    uniprot_list = ['UNIPROT::'+x for x in uniprot_list]
    print(uniprot_list)
    # Steiner subgraph - get a tree using a connected thing...
    tree = explanations.steiner_tree(graph, uniprot_list)
    subgraph = digraph.induced_subgraph([n['name'] for n in tree.vs])
    # use the subgraph to build a connected thing
    # TODO: figure out the rule for combining inputs
    bn_lines = []
    # get all input edges
    for n in subgraph.vs:
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
            elif 'up-regulates' in predicate:
                input_nodes.append(f'({in_node})')
        input_nodes_string = joiner.join(input_nodes)
        if len(input_nodes) == 0:
            input_nodes_string = gene_name
        output_string = f'{gene_name} = {input_nodes_string}'
        bn_lines.append(output_string)
    return '\n'.join(bn_lines)

# TODO: add new genes to an existing boolean network using signor
def add_genes_to_network(bn, gene_list, input_format='symbol', joiner='|'):
    """
    Creates an augmented boolean network from SigNOR using all of the genes along with the genes already present in bn. Tries to build a connected Steiner subgraph...

    Args:
        bn - a BooleanNetwork object
        gene_list - list of gene symbols, gene ids, or uniprot ids.
        input_format - "symbol", "id", or "uniprot"
        joiner - "&" or "|"
    """
    if ' ' not in joiner:
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
        uniprot_list = gene_names.gene_ids_to_uniprot(id_list)
    elif input_format == 'id' or input_format == 'gene_id':
        uniprot_list = gene_names.gene_ids_to_uniprot(gene_list)
    else:
        uniprot_list = gene_list
    # get existing genes
    bn_genes = bn.nodeDict.keys()
    if isinstance(bn_genes[0], str):
        id_list = gene_names.get_ids(bn_genes)
        uniprot_list += gene_names.gene_ids_to_uniprot(id_list)
    uniprot_list = ['UNIPROT::'+x for x in uniprot_list]
    print(uniprot_list)
    # TODO: get a subgraph using the new genes + existing genes
    tree = explanations.steiner_tree(graph, uniprot_list)
    subgraph = digraph.induced_subgraph([n['name'] for n in tree.vs])
    bn_lines = []
    # for each node...
    for n in subgraph.vs:
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
            elif 'up-regulates' in predicate:
                input_nodes.append(f'({in_node})')
        input_nodes_string = joiner.join(input_nodes)
        if len(input_nodes) == 0:
            input_nodes_string = gene_name
        output_string = f'{gene_name} = {input_nodes_string}'
        bn_lines.append(output_string)
