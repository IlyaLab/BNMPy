
import igraph as ig
from pyvis.network import Network
import re

def read_logic_rules(source):
    """
    Reads logic rules from a file path or from a string containing rules.

    Args:
        source (str): Path to the file or the string containing logic rules.

    Returns:
        dict: Mapping from variable names to their logic rules.
    """
    logic_rules = {}
    try:
        # Try to open as a file
        with open(source, 'r') as f:
            lines = f.readlines()
    except (OSError, TypeError):
        # If not a file, treat as string
        lines = source.splitlines()

    for line in lines:
        if line.startswith('#'):
            continue
        parts = line.strip().split('=')
        if len(parts) == 2:
            logic_rules[parts[0].strip()] = parts[1].strip()
    return logic_rules


def build_igraph(logic_rules):
    g = ig.Graph(directed=True)  # Ensure the graph is directed
    node_names = set(logic_rules.keys())

    # Also collect all appearing variables
    for rule in logic_rules.values():
        node_names.update(re.findall(r'\b[A-Za-z0-9_]+\b', rule))

    node_names = list(node_names)
    g.add_vertices(node_names)

    # Add directed edges
    for target, rule in logic_rules.items():
        inputs = set(re.findall(r'\b[A-Za-z0-9_]+\b', rule))
        for input_node in inputs:
            #if input_node != target:
            g.add_edge(input_node, target, label=rule)
    return g

def vis_logic_graph(logic_rules, output_html = "logic_graph.html"):
    """
    Visualize the logic graph using PyVis and igraph.

    Args:
        logic_rules (dict): Dictionary of logic rules.
        output_html (str): Output HTML file name.
    """
    # Check if logic_rules is empty
    if not logic_rules:
        print("No logic rules provided.")
        return

    # Build the graph from rules
    net = Network(notebook=True, directed=True, height='1000px', width='1200px') # Enable directed graph for arrows

    
        

    g = build_igraph(logic_rules)

    # Add nodes to the PyVis network
    for v in g.vs:
        upstream_count = len([pred for pred in g.predecessors(v.index) if pred != v.index])  # Exclude self-predecessors
        downstream_count = len([succ for succ in g.successors(v.index) if succ != v.index])  # Exclude self-successors
        
        # Set color based on upstream and downstream node counts
        if upstream_count == 0:
            color = "grey"  # Source node
        elif downstream_count == 0:
            color = "orange"  # Sink node
        else:
            color = "lightblue"  # Intermediate node

        net.add_node(v.index, label=v["name"], title=v["name"], color=color, size=40, font_size=40, labelHighlightBold=True, physics=True)

    # Add edges to the PyVis network
    for e in g.es:
        src, tgt = e.tuple

        rule_label = e["label"]
        
        if rule_label.startswith("!(") and g.vs[src]["name"] in rule_label:
            # Handle the case where the rule starts with "!("
            net.add_edge(src, tgt, title=rule_label, color="grey", arrowStrikethrough=False)
        elif rule_label.startswith("! (") and g.vs[src]["name"] in rule_label:
            # Handle the case where the rule starts with "!"
            net.add_edge(src, tgt, title=rule_label, color="grey", arrowStrikethrough=False)
        elif "! (" in rule_label and g.vs[src]["name"] in rule_label:
            # Handle the case where the rule contains "! ("
            net.add_edge(src, tgt, title=rule_label, color="grey", arrowStrikethrough=False)
        elif f"!{g.vs[src]['name']}" in rule_label or f"! {g.vs[src]['name']}" in rule_label:
            # Add a separate edge for negation, considering direction
            net.add_edge(src, tgt, title=rule_label, color="blue", arrowStrikethrough=False)
        else:
            net.add_edge(src, tgt, title=rule_label, color="red", arrowStrikethrough=False)

    # Enable physics (for dynamic layout & dragging)
    net.toggle_physics(True)

    # Show the graph
    net.show(output_html)
