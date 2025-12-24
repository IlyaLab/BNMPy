
import igraph as ig
import re
import numpy as np
import json
import tempfile
import os
from pyvis.network import Network
import matplotlib.pyplot as plt
import networkx as nx

def is_entire_rule_negated(rule):
    """
    Check if the entire rule is a single negated expression like !(A | B).
    
    Returns True for rules like:
        - !(A | B)
        - !(A & B & C)
        - ! (A | B)
    
    Returns False for rules like:
        - !(A) | B  (negation is only partial)
        - A & !B    (negation is on individual variable)
        - !A        (no parentheses)
    """
    rule = rule.strip()
    
    # Check for !( or ! (
    if rule.startswith("!("):
        start_idx = 1
    elif rule.startswith("! ("):
        start_idx = 2
    else:
        return False
    
    # Check if opening paren's matching close paren is at the end
    depth = 0
    for i in range(start_idx, len(rule)):
        if rule[i] == '(':
            depth += 1
        elif rule[i] == ')':
            depth -= 1
            if depth == 0:
                # Found the matching closing paren
                return i == len(rule) - 1
    return False


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


def extract_logic_rules_from_network(network):
    """
    Extract logic rules from a BooleanNetwork or PBN object using stored equations.
    
    Args:
        network: BooleanNetwork or ProbabilisticBN object
        
    Returns:
        dict: Mapping from variable names to their logic rules (for BN) or list of rules (for PBN)
        dict: Mapping from edges to probabilities (for PBN) or empty dict (for BN)
    """
    logic_rules = {}
    edge_probabilities = {}
    
    if hasattr(network, 'equations') and network.equations:
        if hasattr(network, 'cij') and hasattr(network, 'gene_functions'):
            for equation in network.equations:
                if '=' in equation:
                    parts = equation.strip().split('=', 1)
                    if len(parts) == 2:
                        node_name = parts[0].strip()
                        rule = parts[1].strip()
                        
                        # For PBN, collect all rules for each node
                        if node_name not in logic_rules:
                            logic_rules[node_name] = []
                        logic_rules[node_name].append(rule)
            
            # Extract probability information for PBN
            for node_name, node_idx in network.nodeDict.items():
                if node_idx < len(network.nf):
                    num_funcs = network.nf[node_idx]
                    if num_funcs > 1:
                        # Multiple functions - extract probabilities
                        for func_offset in range(num_funcs):
                            if func_offset < len(network.cij[node_idx]):
                                probability = network.cij[node_idx, func_offset]
                                if probability > 0:
                                    # Map probability to specific rule
                                    if node_name in logic_rules and func_offset < len(logic_rules[node_name]):
                                        rule = logic_rules[node_name][func_offset]
                                        edge_probabilities[(node_name, rule)] = probability
                    else:
                        # Single function - probability is 1.0
                        if node_name in logic_rules and len(logic_rules[node_name]) > 0:
                            rule = logic_rules[node_name][0]
                            edge_probabilities[(node_name, rule)] = 1.0
        else:
            # This is a BN - use stored equations directly
            for equation in network.equations:
                if '=' in equation:
                    parts = equation.strip().split('=', 1)
                    if len(parts) == 2:
                        node_name = parts[0].strip()
                        rule = parts[1].strip()
                        logic_rules[node_name] = rule
    else:
        print("No logic rules provided.")
    
    return logic_rules, edge_probabilities


def build_igraph_pbn(logic_rules, edge_probabilities):
    """Build igraph for PBN with multiple rules per node."""
    g = ig.Graph(directed=True)
    node_names = set()
    
    # Collect all node names
    for node_name, rules in logic_rules.items():
        node_names.add(node_name)
        if isinstance(rules, list):
            for rule in rules:
                node_names.update(re.findall(r'\b[A-Za-z0-9_]+\b', rule))
        else:
            node_names.update(re.findall(r'\b[A-Za-z0-9_]+\b', rules))
    
    node_names = list(node_names)
    g.add_vertices(node_names)
    
    # Add edges for each rule
    for node_name, rules in logic_rules.items():
        if isinstance(rules, list):
            # PBN with multiple rules
            for rule in rules:
                inputs = set(re.findall(r'\b[A-Za-z0-9_]+\b', rule))
                prob = edge_probabilities.get((node_name, rule), 1.0)
                for input_node in inputs:
                    g.add_edge(input_node, node_name, label=rule, probability=prob)
        else:
            # BN with single rule
            inputs = set(re.findall(r'\b[A-Za-z0-9_]+\b', rules))
            for input_node in inputs:
                g.add_edge(input_node, node_name, label=rules, probability=1.0)
    
    return g


def build_igraph(logic_rules):
    """Build igraph for BN with single rule per node."""
    g = ig.Graph(directed=True)
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
            g.add_edge(input_node, target, label=rule)
    return g


def create_matplotlib_visualization(logic_rules, removed_nodes=None, removed_edges=None, 
                                  measured_nodes=None, perturbed_nodes=None, color_node=None):
    """
    Create a matplotlib-based visualization for Jupyter notebooks.
    
    Args:
        color_node (str): If provided, all nodes will use this color (overrides default coloring)
    """
    removed_nodes = removed_nodes or set()
    removed_edges = removed_edges or set()
    measured_nodes = measured_nodes or set()
    perturbed_nodes = perturbed_nodes or set()
    
    # Create networkx graph
    G = nx.DiGraph()
    
    # Add all nodes mentioned in rules
    all_nodes = set()
    if isinstance(list(logic_rules.values())[0], list):
        # PBN case
        for node_name, rules in logic_rules.items():
            all_nodes.add(node_name)
            for rule in rules:
                all_nodes.update(re.findall(r'\b[A-Za-z0-9_]+\b', rule))
    else:
        # BN case
        all_nodes = set(logic_rules.keys())
        for rule in logic_rules.values():
            all_nodes.update(re.findall(r'\b[A-Za-z0-9_]+\b', rule))
    
    # Add nodes with attributes
    for node in all_nodes:
        G.add_node(node, removed=node in removed_nodes)
    
    # Add edges
    for target, rules in logic_rules.items():
        if isinstance(rules, list):
            # PBN case
            for rule in rules:
                inputs = set(re.findall(r'\b[A-Za-z0-9_]+\b', rule))
                for input_node in inputs:
                    if input_node != target:
                        G.add_edge(input_node, target, rule=rule, removed=(input_node, target) in removed_edges)
        else:
            # BN case
            inputs = set(re.findall(r'\b[A-Za-z0-9_]+\b', rules))
            for input_node in inputs:
                if input_node != target:
                    G.add_edge(input_node, target, rule=rules, removed=(input_node, target) in removed_edges)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # First, separate nodes into removed and normal nodes
    removed_node_list = [n for n in G.nodes() if n in removed_nodes]
    normal_nodes = [n for n in G.nodes() if n not in removed_nodes]
    
    # Categorize nodes based on provided sets and connectivity
    perturbed_node_list = [n for n in normal_nodes if n in perturbed_nodes]
    measured_node_list = [n for n in normal_nodes if n in measured_nodes and n not in perturbed_nodes]
    input_nodes = []
    output_nodes = []
    intermediate_nodes = []
    
    for node in normal_nodes:
        # Skip if already categorized as perturbed or measured
        if node in perturbed_nodes or node in measured_nodes:
            continue
            
        # Check if it's an input node (self-referential rule or no incoming edges)
        is_input = False
        
        if isinstance(logic_rules, dict):
            if isinstance(logic_rules.get(node), str):
                rule = logic_rules[node]
                if rule.strip() == node:
                    is_input = True
        
        # Also consider nodes with no incoming edges as inputs
        if G.in_degree(node) == 0:
            is_input = True
        
        # Check if node is output
        is_output = False
        if not measured_nodes:
            if G.out_degree(node) == 0 and not is_input:
                is_output = True
                

        # Categorize
        if is_input:
            input_nodes.append(node)
        elif is_output:
            output_nodes.append(node)
        else:
            intermediate_nodes.append(node)
    
    # Draw different node types with proper color scheme
    # If a uniform color is provided, use it for all nodes
    if color_node:
        all_normal_nodes = input_nodes + output_nodes + intermediate_nodes + perturbed_node_list + measured_node_list
        if all_normal_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=all_normal_nodes, node_color=color_node, 
                                 node_size=500, alpha=0.8, edgecolors='black')
        if removed_node_list:
            nx.draw_networkx_nodes(G, pos, nodelist=removed_node_list, node_color='lightgrey', 
                                 node_size=500, alpha=0.6, label='Removed', edgecolors='black')
    else:
        if input_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='lightgreen', 
                                 node_size=500, alpha=0.8, label='Input', edgecolors='black')
        if output_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color='yellow', 
                                 node_size=500, alpha=0.8, label='Output', edgecolors='black')
        if intermediate_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=intermediate_nodes, node_color='lightblue', 
                                 node_size=500, alpha=0.8, label='Intermediate', edgecolors='black')
        if removed_node_list:
            nx.draw_networkx_nodes(G, pos, nodelist=removed_node_list, node_color='lightgrey', 
                                 node_size=500, alpha=0.6, label='Removed', edgecolors='black')
        if perturbed_node_list:
            nx.draw_networkx_nodes(G, pos, nodelist=perturbed_node_list, node_color='red', 
                                 node_size=500, alpha=0.8, label='Perturbed', edgecolors='black')
        if measured_node_list:
            nx.draw_networkx_nodes(G, pos, nodelist=measured_node_list, node_color='orange', 
                                 node_size=500, alpha=0.8, label='Measured', edgecolors='black')
    # Draw edges
    normal_edges = [(u, v) for u, v in G.edges() if (u, v) not in removed_edges]
    removed_edge_list = [(u, v) for u, v in G.edges() if (u, v) in removed_edges]
    
    if normal_edges:
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='black', 
                             arrows=True, arrowsize=20, alpha=0.6)
    if removed_edge_list:
        nx.draw_networkx_edges(G, pos, edgelist=removed_edge_list, edge_color='gray', 
                             arrows=True, arrowsize=20, alpha=0.3, style='dashed')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title("Boolean Network Visualization", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Add legend if there are different node types (not when uniform color is used)
    if not color_node and (perturbed_node_list or measured_node_list or input_nodes or intermediate_nodes or output_nodes or removed_node_list):
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.show()
    return None


def vis_network(source, output_html="network_graph.html", interactive=False, 
                   removed_nodes=None, removed_edges=None, measured_nodes=None, perturbed_nodes=None,
                   color_node=None, color_edge=None, physics=True):
    """
    Visualize the logic graph using PyVis and igraph.

    Args:
        source: Logic rules (dict), file path (str), network string (str), 
                BooleanNetwork object, or ProbabilisticBN object
        output_html (str): Output HTML file name.
        interactive (bool): If True, return the network visualization in interactive html file
        removed_nodes (set): Set of node names that were removed (shown in grey)
        removed_edges (set): Set of edge tuples that were removed (shown in grey)
        measured_nodes (set): Set of node names that are measured (shown in orange)
        perturbed_nodes (set): Set of node names that are perturbed (shown in red)
        color_node (str): If provided, all nodes will use this color (e.g., 'lightblue', '#FF5733')
        color_edge (str): If provided, all edges will use this color. If None, edges are colored
                         by regulation type (blue for inhibition, red for activation)
        physics (bool): If True, enable physics simulation
    """
    # Handle different input types
    if hasattr(source, 'nodeDict'):  # BooleanNetwork or ProbabilisticBN
        logic_rules, edge_probabilities = extract_logic_rules_from_network(source)
    elif isinstance(source, dict):
        logic_rules = source
        edge_probabilities = {}
    else:  # String or file path
        logic_rules = read_logic_rules(source)
        edge_probabilities = {}

    # Check if logic_rules is empty
    if not logic_rules:
        print("No logic rules provided.")
        return

    # Set removed nodes and edges to empty sets if not provided
    removed_nodes = removed_nodes or set()
    removed_edges = removed_edges or set()
    measured_nodes = measured_nodes or set()
    perturbed_nodes = perturbed_nodes or set()

    # For non-interactive mode, use matplotlib visualization
    if not interactive:
        return create_matplotlib_visualization(logic_rules, removed_nodes, removed_edges, 
                                                  measured_nodes, perturbed_nodes, color_node=color_node)

    # For interactive mode, use pyvis
    # Check if this is PBN (multiple rules per node)
    is_pbn = isinstance(list(logic_rules.values())[0], list) if logic_rules else False
    
    if is_pbn:
        # Build graph for PBN
        g = build_igraph_pbn(logic_rules, edge_probabilities)
    else:
        # Build graph for BN
        g = build_igraph(logic_rules)
    
    # Create pyvis network with proper configuration
    net = Network(
        height='1000px', 
        width='1200px', 
        bgcolor='#ffffff',
        font_color='black',
        directed=True
    )
    
    # Set network options
    if physics:
        net.set_options("""
        var options = {
            "configure": {
                "enabled": false
            },
            "edges": {
                "color": {
                    "inherit": true
                },
                "smooth": {
                    "enabled": true,
                    "type": "dynamic"
                }
            },
            "interaction": {
                "dragNodes": true,
                "hideEdgesOnDrag": false,
                "hideNodesOnDrag": false
            },
            "physics": {
                "enabled": true,
                "stabilization": {
                    "enabled": true,
                    "fit": true,
                    "iterations": 1000,
                    "onlyDynamicEdges": false,
                    "updateInterval": 50
                }
            }
        }
        """)
    else:
        net.set_options("""
        var options = {
            "configure": {
                "enabled": false
            },
            "edges": {
                "color": {
                    "inherit": true
                },
                "smooth": {
                    "enabled": true,
                    "type": "dynamic"
                }
            },
            "interaction": {
                "dragNodes": true,
                "hideEdgesOnDrag": false,
                "hideNodesOnDrag": false
            },
            "physics": {
                "enabled": false
            }
        }
        """)

    # Add nodes to the PyVis network
    nodes_added = []
    for v in g.vs:
        node_name = v["name"]
        upstream_count = len([pred for pred in g.predecessors(v.index) if pred != v.index])
        downstream_count = len([succ for succ in g.successors(v.index) if succ != v.index])
        
        # If uniform color is provided, use it for all non-removed nodes
        if color_node and node_name not in removed_nodes:
            node_color = color_node
            font_color = "black"
            nodes_added.append(f"{node_name}")
        # Set color based on removal status and node type
        elif node_name in removed_nodes:
            node_color = "lightgrey"
            font_color = "grey"
            nodes_added.append(f"{node_name} (removed)")
        elif node_name in perturbed_nodes:
            node_color = "red"
            font_color = "white"
            nodes_added.append(f"{node_name} (perturbed)")
        elif node_name in measured_nodes:
            node_color = "orange"
            font_color = "black"
            nodes_added.append(f"{node_name} (measured)")
        else:
            # Check if node is input based on logic rules or connectivity
            is_input = False
            
            # self-referential rule or no incoming edges
            if isinstance(logic_rules.get(node_name), str):
                rule = logic_rules[node_name]
                if rule.strip() == node_name:
                    is_input = True
            elif isinstance(logic_rules.get(node_name), list):
                # For PBN, check if all rules are self-referential
                rules = logic_rules[node_name]
                if all(rule.strip() == node_name for rule in rules):
                    is_input = True
            
            # Also consider nodes with no incoming edges as inputs
            if upstream_count == 0:
                is_input = True
            
            # Check if node is output
            is_output = False
            if not measured_nodes:
                if downstream_count == 0 and not is_input:
                    is_output = True
            
            # Set colors based on node type
            if is_input:
                node_color = "lightgreen"  # Light green for inputs
                font_color = "black"
                nodes_added.append(f"{node_name} (input)")
            elif is_output:
                node_color = "yellow"  # Yellow for outputs
                font_color = "black"
                nodes_added.append(f"{node_name} (output)")
            else:
                node_color = "lightblue"  # Light blue for intermediate nodes
                font_color = "black"
                nodes_added.append(f"{node_name} (intermediate)")

        net.add_node(
            v.index, 
            label=node_name, 
            title=node_name, 
            color=node_color, 
            size=40,
            font={'size': 20, 'color': font_color},
            shape='dot'
        )

    # Add edges to the PyVis network
    edges_added = []
    for e in g.es:
        src, tgt = e.tuple
        src_name = g.vs[src]["name"]
        tgt_name = g.vs[tgt]["name"]
        
        rule_label = e["label"]
        
        # Check if edge was removed
        edge_removed = (src_name, tgt_name) in removed_edges
        
        # Get probability if available
        edge_prob = e["probability"] if "probability" in e.attributes() else 1.0
        
        # Create hover title with rule and probability
        if is_pbn and edge_prob < 1.0:
            edge_title = f"{tgt_name} = {rule_label}, p={edge_prob:.3f}"
        else:
            edge_title = f"{tgt_name} = {rule_label}"
        
        # Set edge color and style based on rule type and removal status
        if edge_removed:
            edge_color = "lightgrey"
            edge_width = 1
            edge_alpha = 0.3
            edges_added.append(f"{src_name}->{tgt_name} (removed)")
        elif color_edge:
            # Use uniform edge color if provided
            edge_color = color_edge
            edge_width = 2
            edge_alpha = edge_prob if is_pbn else 1.0
            edges_added.append(f"{src_name}->{tgt_name}")
        elif is_entire_rule_negated(rule_label) and src_name in rule_label:
            # Entire rule is negated like !(A | B), all sources are direct inhibitors
            edge_color = "blue"
            edge_width = 2
            edge_alpha = edge_prob if is_pbn else 1.0
            edges_added.append(f"{src_name}->{tgt_name} (direct inhibitory)")
        elif f"!{src_name}" in rule_label or f"! {src_name}" in rule_label:
            # Direct negation of specific variable like !A or ! A
            edge_color = "blue"
            edge_width = 2
            edge_alpha = edge_prob if is_pbn else 1.0
            edges_added.append(f"{src_name}->{tgt_name} (direct inhibitory)")
        elif rule_label.startswith("!(") and src_name in rule_label:
            # Partial negation like !(A) | B, source is inside negated part
            edge_color = "grey"
            edge_width = 2
            edge_alpha = edge_prob if is_pbn else 1.0
            edges_added.append(f"{src_name}->{tgt_name} (inhibitory)")
        elif rule_label.startswith("! (") and src_name in rule_label:
            edge_color = "grey" 
            edge_width = 2
            edge_alpha = edge_prob if is_pbn else 1.0
            edges_added.append(f"{src_name}->{tgt_name} (inhibitory)")
        elif "! (" in rule_label and src_name in rule_label:
            edge_color = "grey"
            edge_width = 2
            edge_alpha = edge_prob if is_pbn else 1.0
            edges_added.append(f"{src_name}->{tgt_name} (inhibitory)")
        else:
            edge_color = "red"
            edge_width = 2
            edge_alpha = edge_prob if is_pbn else 1.0
            edges_added.append(f"{src_name}->{tgt_name} (activating)")
        
        # For PBN, adjust edge width and transparency based on probability
        if is_pbn and not edge_removed:
            edge_width = max(1, int(edge_prob * 4))  # Scale width by probability
            # Convert alpha to hex color for transparency
            alpha_hex = format(int(edge_alpha * 255), '02x')
            if edge_color == "red":
                edge_color = f"#ff0000{alpha_hex}"
            elif edge_color == "blue":
                edge_color = f"#0000ff{alpha_hex}"
            elif edge_color == "grey":
                edge_color = f"#808080{alpha_hex}"

        net.add_edge(
            src, 
            tgt, 
            title=edge_title, 
            color=edge_color, 
            width=edge_width
        )

    # Add legend for interactive mode (skip if uniform color is used)
    if not color_node:
        legend_x = -1500  # Position legend to the left
        legend_y_start = -800
        legend_spacing = 120
        
        legend_items = [
            ("Input", "lightgreen", "black"),
            ("Output", "yellow", "black"), 
            ("Intermediate", "lightblue", "black"),
            ("Measured", "orange", "black"),
            ("Perturbed", "red", "white"),
            ("Removed", "lightgrey", "grey")
        ]
        
        for i, (label, legend_color, legend_font_color) in enumerate(legend_items):
            net.add_node(
                f"legend_{i}",
                label=label,
                title=f"{label} nodes",
                color=legend_color,
                size=30,
                shape='dot',
                x=legend_x,
                y=legend_y_start + i * legend_spacing,
                physics=False,  # Keep legend nodes fixed
                font={'size': 14, 'color': legend_font_color}
            )

    # Save the network
    net.save_graph(output_html)
    print(f"Network visualization saved to {output_html}")


def vis_compression(original_network, compressed_network, 
                             compression_info, output_html="compression_comparison.html",
                             interactive=False):
    """
    Visualize the original network with removed/collapsed nodes highlighted.
    
    Args:
        original_network: Original BooleanNetwork or ProbabilisticBN
        compressed_network: Compressed network (not used for visualization)
        compression_info: Dictionary with compression information
        output_html (str): Output HTML file name
        interactive (bool): If True, return network visualization in interactive html file
    """
    # Collect all removed nodes from compression info
    removed_nodes = set()
    
    # Add explicitly removed nodes
    removed_nodes.update(compression_info.get('removed_non_observable', set()))
    removed_nodes.update(compression_info.get('removed_non_controllable', set()))
    
    # Add all removed nodes from the general tracking
    removed_nodes.update(compression_info.get('removed_nodes', set()))
    
    # Add intermediate nodes from collapsed paths
    for path in compression_info.get('collapsed_paths', []):
        if len(path) > 2:
            # All intermediate nodes (exclude first and last)
            removed_nodes.update(path[1:-1])
    
    # Get removed edges
    removed_edges = compression_info.get('removed_edges', set())
    
    measured_nodes = compression_info.get('measured_nodes', set())
    perturbed_nodes = compression_info.get('perturbed_nodes', set())
    
    # Use the original network for visualization with removed nodes marked
    return vis_network(
        original_network, 
        output_html, 
        interactive, 
        removed_nodes=removed_nodes,
        removed_edges=removed_edges,
        measured_nodes=measured_nodes,
        perturbed_nodes=perturbed_nodes
    )


def vis_extension(original_network, extended_network, output_html="network+KG.html", interactive=True,
                  color_node='lightblue', color_edge=None, extension_color_node='#E1F2D0', 
                  extension_color_edge=None, physics=True):
    """
    Visualize the network with KG (Knowledge Graph) extension, highlighting new nodes and edges.
    
    Args:
        original_network: Original BooleanNetwork or ProbabilisticBN
        extended_network: Extended network with additional nodes/edges
        output_html (str): Output HTML file name
        interactive (bool): If True, return network visualization in interactive html file
        color_node (str): Color for nodes from the original network (e.g., 'lightblue', '#FF5733').
                         If None, nodes are colored by type (input/output/intermediate)
        color_edge (str): Color for edges from the original network. If None, edges are colored
                         by regulation type (blue for inhibition, red for activation)
        extension_color_node (str): Color for new nodes present only in KG (e.g., '#E1F2D0')
        extension_color_edge (str): Color for new edges present only in KG. If None, edges are 
                                   colored by regulation type (blue for inhibition, red for activation)
        physics (bool): If True, enable physics simulation
    """
    # Extract logic rules from both networks
    original_rules, _ = extract_logic_rules_from_network(original_network)
    extended_rules, extended_probabilities = extract_logic_rules_from_network(extended_network)
    
    # Helper function to extract nodes from rules
    def extract_nodes_from_rules(rules_dict):
        nodes = set()
        for node_name, rules in rules_dict.items():
            nodes.add(node_name)
            if isinstance(rules, list):
                # PBN case - rules is a list of rule strings
                for rule in rules:
                    nodes.update(re.findall(r'\b[A-Za-z0-9_]+\b', rule))
            else:
                # BN case - rules is a single string
                nodes.update(re.findall(r'\b[A-Za-z0-9_]+\b', rules))
        return nodes
    
    # Helper function to extract edges from rules
    def extract_edges_from_rules(rules_dict):
        edges = set()
        for target, rules in rules_dict.items():
            if isinstance(rules, list):
                # PBN case
                for rule in rules:
                    inputs = set(re.findall(r'\b[A-Za-z0-9_]+\b', rule))
                    for input_node in inputs:
                        if input_node != target:
                            edges.add((input_node, target))
            else:
                # BN case
                inputs = set(re.findall(r'\b[A-Za-z0-9_]+\b', rules))
                for input_node in inputs:
                    if input_node != target:
                        edges.add((input_node, target))
        return edges
    
    # Extract nodes and edges
    original_nodes = extract_nodes_from_rules(original_rules)
    extended_nodes = extract_nodes_from_rules(extended_rules)
    original_edges = extract_edges_from_rules(original_rules)
    extended_edges = extract_edges_from_rules(extended_rules)
    
    # Identify new nodes and edges
    new_nodes = extended_nodes - original_nodes
    new_edges = extended_edges - original_edges
    
    print(f"Extension comparison:")
    print(f"  Original nodes: {len(original_nodes)}")
    print(f"  Extended nodes: {len(extended_nodes)}")
    print(f"  New nodes: {len(new_nodes)} - {sorted(new_nodes)}")
    print(f"  Original edges: {len(original_edges)}")
    print(f"  Extended edges: {len(extended_edges)}")
    print(f"  New edges: {len(new_edges)}")
    
    # For non-interactive mode, use matplotlib visualization with highlights
    if not interactive:
        return create_matplotlib_extension_visualization(extended_rules, new_nodes, new_edges,
                                                         color_node=color_node, extension_color_node=extension_color_node,
                                                         extension_color_edge=extension_color_edge)
    
    # For interactive mode, use the extended network but highlight new elements
    # Check if this is PBN
    is_pbn = any(isinstance(rules, list) for rules in extended_rules.values()) if extended_rules else False
    
    if is_pbn:
        # Build graph for PBN
        g = build_igraph_pbn(extended_rules, extended_probabilities)
    else:
        # Build graph for BN
        g = build_igraph(extended_rules)
    
    # Create pyvis network
    net = Network(
        height='1000px', 
        width='1200px', 
        bgcolor='#ffffff',
        font_color='black',
        directed=True
    )
    
    # Set network options
    if physics:
        net.set_options("""
        var options = {
            "configure": {
                "enabled": false
            },
            "edges": {
                "color": {
                    "inherit": true
                },
                "smooth": {
                    "enabled": true,
                    "type": "dynamic"
                }
            },
            "interaction": {
                "dragNodes": true,
                "hideEdgesOnDrag": false,
                "hideNodesOnDrag": false
            },
            "physics": {
                "enabled": true,
                "stabilization": {
                    "enabled": true,
                    "fit": true,
                    "iterations": 1000,
                    "onlyDynamicEdges": false,
                    "updateInterval": 50
                }
            }
        }
        """)
    else:
        net.set_options("""
        var options = {
            "configure": {
                "enabled": false
            },
            "edges": {
                "color": {
                    "inherit": true
                },
                "smooth": {
                    "enabled": true,
                    "type": "dynamic"
                }
            },
            "interaction": {
                "dragNodes": true,
                "hideEdgesOnDrag": false,
                "hideNodesOnDrag": false
            },
            "physics": {
                "enabled": false
            }
        }
        """)

    # Add nodes to the PyVis network
    for v in g.vs:
        node_name = v["name"]
        
        # Check if this is a new node (from KG extension)
        is_new_node = node_name in new_nodes
        
        # Set node color based on whether it's from original network or KG extension
        if is_new_node:
            node_color = extension_color_node
            title = f"{node_name} (from KG)"
        elif color_node:
            node_color = color_node
            title = f"{node_name} (original)"
        else:
            # Default color based on node type
            node_color = "lightblue"
            title = f"{node_name} (original)"
        
        font_color = "black"

        net.add_node(
            v.index, 
            label=node_name, 
            title=title, 
            color=node_color,
            size=40,
            font={'size': 20, 'color': font_color},
            shape='dot'
        )

    # Add edges to the PyVis network
    for e in g.es:
        src, tgt = e.tuple
        src_name = g.vs[src]["name"]
        tgt_name = g.vs[tgt]["name"]
        
        rule_label = e["label"]
        
        # Check if this is a new edge
        is_new_edge = (src_name, tgt_name) in new_edges
        
        # Get probability if available
        edge_prob = e["probability"] if "probability" in e.attributes() else 1.0
        
        # Create hover title with rule and probability
        if is_pbn and edge_prob < 1.0:
            edge_title = f"{tgt_name} = {rule_label}, p={edge_prob:.3f}"
        else:
            edge_title = f"{tgt_name} = {rule_label}"
        
        if is_new_edge:
            edge_title += " (from KG)"
        
        # Determine if this is an inhibitory edge based on rule
        is_inhibitory = (
            (is_entire_rule_negated(rule_label) and src_name in rule_label) or
            f"!{src_name}" in rule_label or 
            f"! {src_name}" in rule_label or
            (rule_label.startswith("!(") and src_name in rule_label) or
            (rule_label.startswith("! (") and src_name in rule_label) or
            ("! (" in rule_label and src_name in rule_label)
        )
        
        # Set edge color and style
        if is_new_edge:
            # New edge from KG
            if extension_color_edge:
                edge_color = extension_color_edge
            else:
                # Fall back to regulation type
                edge_color = "blue" if is_inhibitory else "red"
            edge_width = 3
        else:
            # Original network edge
            if color_edge:
                edge_color = color_edge
            else:
                # Fall back to regulation type
                edge_color = "blue" if is_inhibitory else "red"
            edge_width = 2
        
        # For PBN, adjust edge width and transparency based on probability
        if is_pbn and not is_new_edge and edge_prob < 1.0:
            edge_width = max(1, int(edge_prob * 4))  # Scale width by probability
            # Convert alpha to hex color for transparency
            alpha_hex = format(int(edge_prob * 255), '02x')
            if edge_color == "red":
                edge_color = f"#ff0000{alpha_hex}"
            elif edge_color == "blue":
                edge_color = f"#0000ff{alpha_hex}"
            elif edge_color == "grey":
                edge_color = f"#808080{alpha_hex}"

        net.add_edge(
            src, 
            tgt, 
            title=edge_title, 
            color=edge_color, 
            width=edge_width
        )

    # Add legend
    legend_x = -1500  
    legend_y_start = -800
    legend_spacing = 120
    
    # Build legend items based on what colors are being used
    legend_items = []
    if color_node:
        legend_items.append(("Original Network", color_node, "black"))
    legend_items.append(("KG Extension (Node)", extension_color_node, "black"))
    if extension_color_edge:
        legend_items.append(("KG Extension (Edge)", extension_color_edge, "black"))
    
    for i, (label, legend_color, font_color) in enumerate(legend_items):
        net.add_node(
            f"legend_{i}",
            label=label,
            title=f"{label}",
            color=legend_color,
            size=30,
            shape='dot',
            x=legend_x,
            y=legend_y_start + i * legend_spacing,
            physics=False,
            font={'size': 14, 'color': font_color}
        )

    # Save the network
    net.save_graph(output_html)
    print(f"Extension visualization saved to {output_html}")


def create_matplotlib_extension_visualization(logic_rules, new_nodes, new_edges,
                                              color_node='lightblue', extension_color_node='#E1F2D0',
                                              extension_color_edge=None):
    """
    Create a matplotlib-based visualization for extension comparison.
    
    Args:
        logic_rules: Dictionary of logic rules
        new_nodes: Set of new node names from KG extension
        new_edges: Set of new edge tuples from KG extension
        color_node (str): Color for nodes from the original network
        extension_color_node (str): Color for new nodes from KG
        extension_color_edge (str): Color for new edges from KG. If None, uses default color
    """
    # Create networkx graph
    G = nx.DiGraph()
    
    # Add all nodes mentioned in rules
    all_nodes = set()
    for node_name, rules in logic_rules.items():
        all_nodes.add(node_name)
        if isinstance(rules, list):
            # PBN case
            for rule in rules:
                all_nodes.update(re.findall(r'\b[A-Za-z0-9_]+\b', rule))
        else:
            # BN case
            all_nodes.update(re.findall(r'\b[A-Za-z0-9_]+\b', rules))
    
    # Add nodes with attributes
    for node in all_nodes:
        G.add_node(node, is_new=node in new_nodes)
    
    # Add edges
    for target, rules in logic_rules.items():
        if isinstance(rules, list):
            # PBN case
            for rule in rules:
                inputs = set(re.findall(r'\b[A-Za-z0-9_]+\b', rule))
                for input_node in inputs:
                    if input_node != target:
                        G.add_edge(input_node, target, is_new=(input_node, target) in new_edges)
        else:
            # BN case
            inputs = set(re.findall(r'\b[A-Za-z0-9_]+\b', rules))
            for input_node in inputs:
                if input_node != target:
                    G.add_edge(input_node, target, is_new=(input_node, target) in new_edges)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Use spring layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Separate new and existing nodes
    existing_nodes = [n for n in G.nodes() if n not in new_nodes]
    new_node_list = [n for n in G.nodes() if n in new_nodes]
    
    # Draw nodes with custom colors
    original_node_color = color_node if color_node else 'lightblue'
    if existing_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=existing_nodes, node_color=original_node_color, 
                             node_size=500, alpha=0.8, label='Original Network', edgecolors='black')
    if new_node_list:
        nx.draw_networkx_nodes(G, pos, nodelist=new_node_list, node_color=extension_color_node, 
                             node_size=500, alpha=0.8, label='KG Extension', edgecolors='black')
    
    # Draw edges
    existing_edges = [(u, v) for u, v in G.edges() if (u, v) not in new_edges]
    new_edge_list = [(u, v) for u, v in G.edges() if (u, v) in new_edges]
    
    if existing_edges:
        nx.draw_networkx_edges(G, pos, edgelist=existing_edges, edge_color='black', 
                             arrows=True, arrowsize=20, alpha=0.6)
    if new_edge_list:
        new_edge_color = extension_color_edge if extension_color_edge else 'orange'
        nx.draw_networkx_edges(G, pos, edgelist=new_edge_list, edge_color=new_edge_color, 
                             arrows=True, arrowsize=20, alpha=0.8, width=3)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title("Network + KG Extension Visualization", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.show()
    return None
