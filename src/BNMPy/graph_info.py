# TODO: get info on available graphs
import os
import random
import zipfile

import igraph as ig
import numpy as np
import pandas as pd

PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PATH, 'KG_files')
MSIGDB_PATH = os.path.join(PATH, 'raw_graphs/msigdb_v2023.1.Hs_json_files_to_download_locally.zip')

def get_available_graphs():
    files = os.listdir(DATA_PATH)
    return files

def get_available_msigdb():
    f = zipfile.ZipFile(MSIGDB_PATH)
    files = f.namelist()
    f.close()
    return files


def load_graph(filename):
    files = os.listdir(DATA_PATH)
    if filename not in files:
        # if filename not in files, try opening it as a path
        if not os.path.exists(filename):
            raise FileNotFoundError()
    else:
        filename = os.path.join(DATA_PATH, filename)
    if 'tsv' in filename:
        df = pd.read_csv(filename, sep='\t')
    else:
        df = pd.read_csv(filename)
    return df


def _load_msigdb(data, object_category='BiologicalProcess', object_id_prefix='MSigDB',
        predicate='participates_in'):
    """
    Returns a pandas edge table and a graph
    """
    import gene_names
    new_entries = []
    for k, v in data.items():
        gene_set_name = k
        source = v['exactSource']
        ref = v['pmid']
        genes = v['geneSymbols']
        try:
            gene_ids = gene_names.get_ids(genes)
        except:
            gene_ids = []
            for g in genes:
                try:
                    gene_id = gene_names.get_ids([g])
                    gene_ids.append(gene_id)
                except:
                    continue
        for gene_id, symbol in zip(gene_ids, genes):
            entry = {}
            entry['subject_category'] = 'Gene'
            entry['subject_id_prefix'] = 'NCBIGene'
            entry['subject_id'] = gene_id
            entry['subject_name'] = symbol
            entry['predicate'] = predicate
            entry['object_category'] = object_category
            entry['object_id_prefix'] = object_id_prefix
            entry['object_id'] = gene_set_name
            entry['object_name'] = gene_set_name
            entry['Primary_Knowledge_Source'] = source
            entry['Knowledge_Source'] = v['collection']
            entry['publications'] = ref
            new_entries.append(entry)
    return pd.DataFrame(new_entries)


def load_msigdb(name):
    """Load an msigdb graph as a dataframe. Input name is one of the filenames returned by get_available_msigdb."""
    import json
    f = zipfile.ZipFile(MSIGDB_PATH)
    m = f.getinfo(name)
    json_file = f.open(m)
    data = json.load(json_file)
    df = _load_msigdb(data)
    f.close()
    return df

def df_to_networkx(df, directed=False):
    """
    Converts a panda dataframe to a networkx Graph (or DiGraph), with node and edge attributes.
    """
    import networkx as nx
    create_using = nx.Graph
    if directed:
        create_using = nx.DiGraph
    df['subject_id_full'] = df['subject_id_prefix'] + '::' + df['subject_id'].astype(str)
    df['object_id_full'] = df['object_id_prefix'] + '::' + df['object_id'].astype(str)
    graph = nx.from_pandas_edgelist(df, source='subject_id_full', target='object_id_full',
            edge_attr=['predicate',
                       'Primary_Knowledge_Source',
                       'Knowledge_Source',
                       'publications'],
            create_using=create_using)
    node_attributes = {}
    for _, row in df.iterrows():
        if row['subject_id'] not in node_attributes:
            node_attributes[row['subject_id_full']] = {
                    'id': row['subject_id'],
                    'id_prefix': row['subject_id_prefix'],
                    'name': row['subject_name'],
                    'category': row['subject_category']}
        if row['object_id'] not in node_attributes:
            node_attributes[row['object_id_full']] = {
                    'id': row['object_id'],
                    'id_prefix': row['object_id_prefix'],
                    'name': row['object_name'],
                    'category': row['object_category']}
    nx.set_node_attributes(graph, node_attributes)
    return graph

def df_to_graph(df, directed=False):
    """
    Converts a panda dataframe to an igraph.Graph (or DiGraph), with node and edge attributes.
    """
    df['subject_id_full'] = df['subject_id_prefix'] + '::' + df['subject_id'].astype(str)
    df['object_id_full'] = df['object_id_prefix'] + '::' + df['object_id'].astype(str)
    # reorder?
    edges_df = df[['subject_id_full', 'object_id_full', 'predicate', 'Primary_Knowledge_Source', 'Knowledge_Source', 'publications', 'score']]
    graph = ig.Graph.DataFrame(edges_df, directed=directed, use_vids=False)
    node_attributes = {}
    for _, row in df.iterrows():
        if row['subject_id'] not in node_attributes:
            node_attributes[row['subject_id_full']] = {
                    'id': row['subject_id'],
                    'id_prefix': row['subject_id_prefix'],
                    'feature_name': row['subject_name'],
                    'category': row['subject_category']}
        if row['object_id'] not in node_attributes:
            node_attributes[row['object_id_full']] = {
                    'id': row['object_id'],
                    'id_prefix': row['object_id_prefix'],
                    'feature_name': row['object_name'],
                    'category': row['object_category']}
    for v in graph.vs:
        attributes = node_attributes[v['name']]
        v.update_attributes(attributes)
    return graph


def get_nodes_table(graph):
    """
    Returns a Pandas DataFrame of the nodes.
    """
    rows = []
    for v in graph.vs:
        row = {'id': v['name']}
        row.update(v.attributes())
        rows.append(row)
    return pd.DataFrame(rows)

def get_names_to_ids(graph, category=None):
    """Returns a dict mapping node names to IDs (ignoring prefixes and categories so on)"""
    names_to_ids = {}
    for v in graph.vs:
        if category is not None and v['category'] != category:
            continue
        names_to_ids[v['feature_name']] = v['name']
    return names_to_ids

def get_names_to_ids_networkx(graph, category=None):
    """Returns a dict mapping node names to IDs (ignoring prefixes and categories so on)"""
    names_to_ids = {}
    for n, attrs in graph.nodes.items():
        if category is not None and attrs['category'] != category:
            continue
        names_to_ids[attrs['name']] = n
    return names_to_ids

def get_spoke_categories(graph):
    return set(attrs['category'] for attrs in graph.vs)

def get_spoke_sources(graph):
    return set(attrs['source'] for attrs in graph.vs)

def spoke_identifiers_to_ids(graph, category, source=None):
    """
    Returns a mapping from SPOKE identifiers to IDs.

    category: 'Protein', 'Gene', 'Compound', 'Disease', etc
    source: 'KEGG', ...
    """
    identifiers_to_ids = {}
    for v in graph.vs:
        attrs = v.attributes()
        if 'category' in attrs and attrs['category'] == category:
            if source is None or ('source' in attrs and attrs['source'] == source):
                identifiers_to_ids[attrs['identifier']] = v['name']
    return identifiers_to_ids

def spoke_identifiers_to_ids_networkx(graph, category, source=None):
    """
    Returns a mapping from SPOKE identifiers to IDs.

    category: 'Protein', 'Gene', 'Compound', 'Disease', etc
    source: 'KEGG', ...
    """
    identifiers_to_ids = {}
    for n, attrs in graph.nodes.items():
        if 'category' in attrs and attrs['category'] == category:
            if source is None or ('source' in attrs and attrs['source'] == source):
                identifiers_to_ids[attrs['identifier']] = n
    return identifiers_to_ids

def get_category_ids_to_nodes(graph, category):
    """
    Returns a dict that maps from identifiers in the specified category to graph node indices, for graphs that are not SPOKE.
    """
    identifiers_to_ids = {}
    for v in graph.vs:
        if v['category'] == category:
            identifiers_to_ids[v['id']] = v.index 
    return identifiers_to_ids

def largest_component(graph):
    "Returns a subgraph containing the largest connected component of the given graph."
    components = graph.connected_components()
    sizes = components.sizes()
    largest_component = 0
    largest_component_size = 0
    for i, c in enumerate(sizes):
        if c > largest_component_size:
            largest_component_size = c
            largest_component = i
    subgraph = components.subgraph(largest_component)
    return subgraph


def nodes_in_category(graph, category, attr_name='category'):
    "Returns all nodes that are within a given category, as a list of igraph.Vertex objects."
    nodes_in_category = []
    for v in graph.vs:
        attrs = v.attributes()
        if attr_name in attrs and attrs[attr_name] == category:
            nodes_in_category.append(v)
    return nodes_in_category


def nodes_in_category_networkx(graph, category, attr_name='category'):
    nodes_in_category = []
    for n, attrs in graph.nodes.items():
        if attr_name in attrs and attrs[attr_name] == category:
            nodes_in_category.append(n)
    return nodes_in_category

def random_nodes_in_category(graph, category, n_nodes):
    """
    Returns a list of random node ids in the given category.
    """
    nodes_in_category = []
    for v in graph.vs:
        attrs = v.attributes()
        if 'category' in attrs and attrs['category'] == category:
            nodes_in_category.append(attrs['name'])
    return random.sample(nodes_in_category, n_nodes)

def random_nodes(graph, n_nodes):
    """
    Returns a list of random node ids.
    """
    return random.sample([v['name'] for v in graph.vs], n_nodes)


def random_nodes_in_category_networkx(graph, category, n_nodes):
    """
    Returns a list of random spoke ids in the given category.
    """
    nodes_in_category = []
    for n, attrs in graph.nodes.items():
        if 'category' in attrs and attrs['category'] == category:
            nodes_in_category.append((n, attrs['identifier']))
    return random.sample(nodes_in_category, n_nodes)


# random nodes with similar degree distributions? investigative bias - constrain null model to be similar to the problem. We could select random nodes among the nodes that are in the general set...
def degree_sample(graph, node_list, n_samples, dist_or_probs):
    """
    Degree-based node sampling, to sample nodes such that they approximately match the given degree distribution.

    Args:
        graph - an igraph.Graph
        node_list - a list of vertices to be sampled from
        n_samples - the number of points to sample
        dist_or_probs - Either a scipy.stats distribution that has a pdf function (could probably use a kernel density estimation), or a list/array of probabilities over node_list
    """
    prob_vals = []
    # lol 
    if hasattr(dist_or_probs, '__getitem__'):
        prob_vals = dist_or_probs
    else:
        dist = dist_or_probs
        prob_vals = dist.pdf(graph.degree(node_list))
        prob_vals = np.array(prob_vals)
        prob_vals = prob_vals/prob_vals.sum()
    sampled_nodes = np.random.choice(node_list, size=n_samples, replace=False, p=prob_vals)
    return sampled_nodes
