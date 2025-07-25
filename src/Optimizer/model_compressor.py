import numpy as np
import networkx as nx
from typing import Set, List, Dict, Tuple, Union, Optional
from collections import deque
import copy
import re


class ModelCompressor:
    """
    Model compression for Boolean Networks.
    
    Provides methods to compress models by removing non-observable/non-controllable
    nodes and collapsing linear paths to simplify the network structure.
    """
    
    # Regex pattern to match single identifier (alias) rules
    _alias_pat = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
    
    def __init__(self, network, measured_nodes: Set[str] = None, perturbed_nodes: Set[str] = None):
        """
        Initialize the model compressor.
        
        Parameters:
        -----------
        network : BooleanNetwork
            The Boolean network to compress
        measured_nodes : Set[str], optional
            Set of node names that are measured/observed. If None, will auto-detect output nodes.
        perturbed_nodes : Set[str], optional  
            Set of node names that are perturbed/controlled
        """
        # Check if it's a PBN and warn user
        if hasattr(network, 'cij'):
            raise ValueError("PBN compression is not currently supported. Please use Boolean Networks only.")
        
        self.network = network
        self.perturbed_nodes = perturbed_nodes or set()
        
        # Build directed graph representation
        self.graph = self._build_graph()
        
        # Auto-detect measured nodes if not provided
        if measured_nodes is None:
            self.measured_nodes = self._detect_output_nodes()
        else:
            self.measured_nodes = measured_nodes
        
        # Track original network structure for restoration
        self.original_structure = self._save_network_structure()
        
        # Track removed elements for visualization
        self.removed_nodes = set()
        self.collapsed_paths = []
        
    def _build_graph(self) -> nx.DiGraph:
        """
        Build a NetworkX directed graph from the Boolean network connectivity matrix.
        
        Returns:
        --------
        nx.DiGraph
            Directed graph representation of the network
        """
        graph = nx.DiGraph()
        
        # Add all nodes
        for name, idx in self.network.nodeDict.items():
            graph.add_node(name, index=idx)
        
        # Add edges based on connectivity matrix for BooleanNetwork
        for node_idx in range(self.network.N):
            for j in range(self.network.K[node_idx]):
                input_idx = self.network.varF[node_idx, j]
                if input_idx != -1:
                    input_name = self._get_node_name(input_idx)
                    output_name = self._get_node_name(node_idx)
                    if input_name and output_name:
                        graph.add_edge(input_name, output_name)
        
        return graph
    
    def _get_node_name(self, index: int) -> Optional[str]:
        """Get node name from index."""
        for name, idx in self.network.nodeDict.items():
            if idx == index:
                return name
        return None
    
    def _save_network_structure(self) -> Dict:
        """Save current network structure for potential restoration."""
        structure = {
            'varF': self.network.varF.copy(),
            'F': self.network.F.copy(),
            'nodeDict': self.network.nodeDict.copy(),
            'N': self.network.N,
            'K': self.network.K.copy(),
        }
        return structure
    
    def find_non_observable_nodes(self) -> Set[str]:
        """
        Find nodes that are non-observable (no path to measured species).
        
        Returns:
        --------
        Set[str]
            Set of non-observable node names
        """
        if not self.measured_nodes:
            return set()
        
        # Find all nodes that can reach measured nodes
        observable_nodes = set()
        
        for measured_node in self.measured_nodes:
            if measured_node in self.graph:
                # Find all predecessors (nodes that can reach this measured node)
                predecessors = nx.ancestors(self.graph, measured_node)
                observable_nodes.update(predecessors)
                observable_nodes.add(measured_node)  # measured node is observable
        
        # Non-observable nodes are those not in the observable set
        all_nodes = set(self.graph.nodes())
        non_observable = all_nodes - observable_nodes
        
        return non_observable
    
    def find_non_controllable_nodes(self) -> Set[str]:
        """
        Find nodes that are non-controllable (no path from perturbed species).
        
        Returns:
        --------
        Set[str]
            Set of non-controllable node names
        """
        if not self.perturbed_nodes:
            return set()
        
        # Find all nodes reachable from perturbed nodes
        controllable_nodes = set()
        
        for perturbed_node in self.perturbed_nodes:
            if perturbed_node in self.graph:
                # Find all successors (nodes reachable from this perturbed node)
                successors = nx.descendants(self.graph, perturbed_node)
                controllable_nodes.update(successors)
                controllable_nodes.add(perturbed_node)  # perturbed node is controllable
        
        # Non-controllable nodes are those not in the controllable set
        all_nodes = set(self.graph.nodes())
        non_controllable = all_nodes - controllable_nodes
        
        return non_controllable
    
    def find_collapsible_paths(self) -> List[List[str]]:
        """
        Find linear paths that can be collapsed.
        
        A collapsible path is a series of nodes that form a linear cascade,
        where intermediate nodes can be removed without losing connectivity.
        
        Returns:
        --------
        List[List[str]]
            List of paths, where each path is a list of node names
        """
        collapsible_paths = []
        visited = set()
        
        # Important nodes that should not be removed
        important_nodes = self.measured_nodes | self.perturbed_nodes
        complex_nodes = set()
        for node in self.graph.nodes():
            if self._is_used_in_complex_rule(node):
                complex_nodes.add(node)
        
        protected_nodes = important_nodes | complex_nodes
        
        # Find all linear paths starting from any node
        for node in self.graph.nodes():
            if node in visited:
                continue
                
            # Look for the start of a potential path
            path = self._trace_linear_path(node, protected_nodes, visited)
            
            if len(path) > 1:  # Only paths with at least one node to remove
                # Check if this path has collapsible nodes
                collapsible_nodes = []
                for i, n in enumerate(path):
                    if n in protected_nodes:
                        # If this is the last node, it's okay (endpoint)
                        if i == len(path) - 1:
                            break
                        else:
                            # Protected node in the middle - can't collapse this path
                            collapsible_nodes = []
                            break
                    else:
                        collapsible_nodes.append(n)
                
                if collapsible_nodes:
                    collapsible_paths.append(path)
                    visited.update(path[:-1] if path[-1] in protected_nodes else path)
        
        return collapsible_paths

    def _trace_linear_path(self, start_node: str, important_nodes: Set[str], visited: Set[str]) -> List[str]:
        """
        Trace a linear path starting from the given node.
        
        A linear path is a sequence of nodes where each node has exactly one output
        (except the last) and exactly one input (except the first).
        
        Important nodes and nodes used in complex rules can be endpoints but not intermediate nodes.
        """
        path = [start_node]
        current = start_node
        
        # Follow the path forward
        while True:
            successors = list(self.graph.successors(current))
            
            # Check if we can continue the path
            if len(successors) != 1:
                break
                
            next_node = successors[0]
            
            # Check if next node is suitable for the path
            if (next_node in visited or 
                next_node in path or  # Avoid cycles
                self.graph.has_edge(next_node, next_node)):  # Avoid self-loops
                break
            
            # If we've reached an important node
            if next_node in important_nodes:
                path.append(next_node)
                break
                
            # If we've reached a node used in complex rules
            if self._is_used_in_complex_rule(next_node):
                path.append(next_node)
                break
                
            # Check if current node is linear (for continuing the path)
            if not self._is_linear_node(next_node):
                break
            
            # Add to path and continue
            path.append(next_node)
            current = next_node
        
        return path
    
    def _is_linear_node(self, node: str) -> bool:
        """
        Check if a node is part of a linear cascade.
        
        A linear node has exactly one input and one output,
        and is not involved in self-loops.
        """
        # Check for self-loops
        if self.graph.has_edge(node, node):
            return False
        
        # Check input/output constraints for intermediate nodes
        in_degree = self.graph.in_degree(node)
        out_degree = self.graph.out_degree(node)
        
        # A linear intermediate node has exactly one input and one output
        return in_degree == 1 and out_degree == 1
    
    def collapse_paths(self, paths: List[List[str]]) -> None:
        """
        Collapse linear paths by removing intermediate nodes and creating direct connections.
        For each path, update all references to any relay node in the path to the ultimate source.
        """
        self.collapsed_paths = paths.copy()
        important_nodes = self.measured_nodes | self.perturbed_nodes
        
        # Also consider nodes used in complex rules as important
        nodes_in_complex_rules = set()
        for node in self.graph.nodes():
            if self._is_used_in_complex_rule(node):
                nodes_in_complex_rules.add(node)
        
        all_important_nodes = important_nodes | nodes_in_complex_rules
        
        # Process each path
        for path in paths:
            if len(path) < 2:
                continue
                
            # Get the ultimate source (the node that should replace all others in the path)
            src = self._ultimate_source(path[0])
            
            # Determine which nodes in the path should be removed and which preserved
            nodes_to_replace = []
            preserved_nodes = []
            
            for node in path:
                if node not in all_important_nodes:
                    nodes_to_replace.append(node)
                else:
                    preserved_nodes.append(node)
            
            # Update rules for preserved nodes at path endpoints
            for preserved_node in preserved_nodes:
                # If this preserved node was getting its input from a node in the path,
                # update it to get input from the ultimate source instead
                current_rule = self._get_rule(preserved_node).strip()
                
                # Replace any nodes from this path with the ultimate source
                new_rule = current_rule
                for node in nodes_to_replace:
                    pattern = r'\b' + re.escape(node) + r'\b'
                    new_rule = re.sub(pattern, src, new_rule)
                
                if new_rule != current_rule:
                    self._update_node_rule(preserved_node, new_rule)
            
            # Update all other equations that reference nodes to be replaced
            for node in nodes_to_replace:
                # Update all equations that reference this node
                for i, eq in enumerate(self.network.equations):
                    lhs, rhs = eq.split('=', 1)
                    lhs = lhs.strip()
                    rule = rhs.strip()
                    
                    # Skip equations for nodes we're removing
                    if lhs in nodes_to_replace:
                        continue
                    
                    # Replace this node with the ultimate source
                    pattern = r'\b' + re.escape(node) + r'\b'
                    new_rule = re.sub(pattern, src, rule)
                    
                    if new_rule != rule:
                        self.network.equations[i] = f"{lhs} = {new_rule}"
                        self._update_node_rule(lhs, new_rule)
        
        # Remove all nodes that were marked for removal
        all_nodes_to_remove = set()
        
        # Collect all nodes that should be removed from all paths
        for path in paths:
            if len(path) < 2:
                continue
                
            for node in path:
                if node not in all_important_nodes:
                    all_nodes_to_remove.add(node)
        
        # Update self.removed_nodes with the actual nodes being removed
        self.removed_nodes.update(all_nodes_to_remove)
        
        if all_nodes_to_remove:
            self.remove_nodes(all_nodes_to_remove)
            # Remove equations for removed nodes
            if hasattr(self.network, 'equations'):
                self.network.equations = [eq for eq in self.network.equations 
                                        if eq.split('=')[0].strip() not in all_nodes_to_remove]
            # Remove nodes from nodeDict
            for node in all_nodes_to_remove:
                if node in self.network.nodeDict:
                    del self.network.nodeDict[node]
        
        self.graph = self._build_graph()
        self._collapse_alias_nodes()
    
    def _update_node_rule(self, node_name: str, new_rule: str) -> None:
        """Update the rule for a specific node and its connectivity matrix."""
        if not hasattr(self.network, 'equations') or not new_rule:
            return
        
        # Update the equations list
        for i, equation in enumerate(self.network.equations):
            eq_parts = equation.split('=', 1)
            if len(eq_parts) == 2 and eq_parts[0].strip() == node_name:
                self.network.equations[i] = f"{node_name} = {new_rule}"
                break
        
        # Update connectivity matrix
        if node_name in self.network.nodeDict:
            node_idx = self.network.nodeDict[node_name]
            
            # Clear existing connections
            for j in range(self.network.varF.shape[1]):
                self.network.varF[node_idx, j] = -1
            
            # Extract new dependencies from the rule
            import re
            dependencies = re.findall(r'\b[A-Za-z0-9_]+\b', new_rule)
            
            # Add new connections
            conn_count = 0
            for dep in dependencies:
                if dep in self.network.nodeDict and dep != node_name:
                    dep_idx = self.network.nodeDict[dep]
                    if conn_count < self.network.varF.shape[1]:
                        self.network.varF[node_idx, conn_count] = dep_idx
                        conn_count += 1
            
            # Update connection count
            self.network.K[node_idx] = conn_count
    
    def remove_nodes(self, nodes_to_remove: Set[str]) -> None:
        """
        Remove specified nodes from the network.
        
        Parameters:
        -----------
        nodes_to_remove : Set[str]
            Set of node names to remove
        """
        if not nodes_to_remove:
            return
        
        # Track removed nodes for visualization
        self.removed_nodes.update(nodes_to_remove)
        
        # Get indices of nodes to remove
        indices_to_remove = [self.network.nodeDict[name] for name in nodes_to_remove 
                           if name in self.network.nodeDict]
        
        if not indices_to_remove:
            return
        
        # Sort in descending order to maintain correct indices when removing
        indices_to_remove.sort(reverse=True)
        
        # Create new nodeDict mapping
        new_nodeDict = {}
        new_index = 0
        
        for name, old_index in self.network.nodeDict.items():
            if name not in nodes_to_remove:
                new_nodeDict[name] = new_index
                new_index += 1
        
        # Update network structure
        self._update_network_structure(indices_to_remove, new_nodeDict)
    
    def _update_network_structure(self, indices_to_remove: List[int], new_nodeDict: Dict[str, int]) -> None:
        """Update network structure after removing nodes."""
        # Create index mapping from old to new
        old_to_new = {}
        new_idx = 0
        for old_idx in range(self.network.N):
            if old_idx not in indices_to_remove:
                old_to_new[old_idx] = new_idx
                new_idx += 1
        
        # Update network properties
        new_N = len(new_nodeDict)
        self._update_bn_structure(indices_to_remove, old_to_new, new_N, new_nodeDict)
    
    def _update_bn_structure(self, indices_to_remove: List[int], old_to_new: Dict[int, int], 
                           new_N: int, new_nodeDict: Dict[str, int]) -> None:
        """Update BooleanNetwork structure."""
        # Create new arrays
        new_varF = []
        new_F = []
        new_K = []
        new_equations = []
        
        for old_idx in range(self.network.N):
            if old_idx not in indices_to_remove:
                # Update connectivity matrix row
                old_row = self.network.varF[old_idx]
                new_row = []
                for conn in old_row:
                    if conn == -1:
                        new_row.append(-1)
                    elif conn in old_to_new:
                        new_row.append(old_to_new[conn])
                    # Skip connections to removed nodes
                
                # Pad with -1 if necessary
                while len(new_row) < self.network.varF.shape[1]:
                    new_row.append(-1)
                
                new_varF.append(new_row[:self.network.varF.shape[1]])
                new_F.append(self.network.F[old_idx])
                
                # Update K (count non-(-1) connections)
                new_K.append(sum(1 for x in new_row if x != -1))
                
                # Update equations if they exist
                if hasattr(self.network, 'equations') and self.network.equations:
                    if old_idx < len(self.network.equations):
                        new_equations.append(self.network.equations[old_idx])
        
        # Update network properties
        self.network.varF = np.array(new_varF)
        self.network.F = np.array(new_F)
        self.network.K = np.array(new_K)
        self.network.N = new_N
        self.network.nodeDict = new_nodeDict
        
        # Update equations if they exist
        if hasattr(self.network, 'equations') and new_equations:
            self.network.equations = new_equations
        
        # Update other arrays if they exist
        if hasattr(self.network, 'nodes'):
            new_nodes = []
            for old_idx in range(len(self.network.nodes)):
                if old_idx not in indices_to_remove:
                    new_nodes.append(self.network.nodes[old_idx])
            self.network.nodes = np.array(new_nodes)
    
    def compress(self, remove_non_observable: bool = True, 
                 remove_non_controllable: bool = True,
                 collapse_linear_paths: bool = True) -> Dict[str, Set[str]]:
        """
        Compress the model by removing non-observable/non-controllable nodes
        and collapsing linear paths.
        
        Parameters:
        -----------
        remove_non_observable : bool, default=True
            Whether to remove non-observable nodes
        remove_non_controllable : bool, default=True  
            Whether to remove non-controllable nodes
        collapse_linear_paths : bool, default=True
            Whether to collapse linear paths
            
        Returns:
        --------
        Dict[str, Set[str]]
            Dictionary containing information about the compression:
            - 'removed_non_observable': Set of removed non-observable nodes
            - 'removed_non_controllable': Set of removed non-controllable nodes  
            - 'collapsed_paths': List of collapsed paths
            - 'removed_nodes': Set of all removed nodes
            - 'removed_edges': Set of all removed edges
        """
        # Store original structure for edge comparison
        original_graph = self._build_graph()
        
        compression_info = {
            'removed_non_observable': set(),
            'removed_non_controllable': set(),
            'collapsed_paths': [],
            'measured_nodes': self.measured_nodes,
            'perturbed_nodes': self.perturbed_nodes
        }
        
        # Find nodes to remove
        nodes_to_remove = set()
        
        if remove_non_observable:
            non_observable = self.find_non_observable_nodes()
            nodes_to_remove.update(non_observable)
            compression_info['removed_non_observable'] = non_observable
        
        if remove_non_controllable:
            non_controllable = self.find_non_controllable_nodes()
            nodes_to_remove.update(non_controllable)
            compression_info['removed_non_controllable'] = non_controllable
        
        # Remove non-observable/non-controllable nodes first
        if nodes_to_remove:
            self.remove_nodes(nodes_to_remove)
            # Rebuild graph after removal
            self.graph = self._build_graph()
        
        # Collapse linear paths
        if collapse_linear_paths:
            paths = self.find_collapsible_paths()
            if paths:
                self.collapse_paths(paths)
                compression_info['collapsed_paths'] = paths
                # Rebuild graph after collapsing
                self.graph = self._build_graph()
        
        self._collapse_alias_nodes()
        
        # Identify removed edges by comparing original and final structures
        compression_info['removed_edges'] = self._identify_removed_edges(original_graph)
        compression_info['removed_nodes'] = self.removed_nodes
        
        return compression_info
    
    def _identify_removed_edges(self, original_graph: nx.DiGraph) -> Set[Tuple[str, str]]:
        """
        Identify removed edges by comparing original and current graph structures.
        
        Parameters:
        -----------
        original_graph : nx.DiGraph
            The original graph before compression
            
        Returns:
        --------
        Set[Tuple[str, str]]
            Set of removed edges as (source, target) tuples
        """
        removed_edges = set()
        
        # Get current graph edges
        current_edges = set(self.graph.edges())
        
        # Find edges that existed in original but not in current
        for edge in original_graph.edges():
            if edge not in current_edges:
                removed_edges.add(edge)
        
        # Also include edges involving removed nodes
        for node in self.removed_nodes:
            # Add all edges from/to removed nodes that were in the original
            for pred in original_graph.predecessors(node):
                removed_edges.add((pred, node))
            for succ in original_graph.successors(node):
                removed_edges.add((node, succ))
        
        return removed_edges
    
    def get_compression_summary(self, compression_info: Dict) -> str:
        """
        Generate a summary of the compression results.
        
        Parameters:
        -----------
        compression_info : Dict
            Compression information returned by compress()
            
        Returns:
        --------
        str
            Human-readable summary of compression
        """
        summary = ["Model Compression Summary:"]
        summary.append("=" * 30)
        
        # Add information about measured and perturbed nodes
        measured_nodes = compression_info.get('measured_nodes', set())
        perturbed_nodes = compression_info.get('perturbed_nodes', set())
        important_nodes = measured_nodes | perturbed_nodes
        
        if measured_nodes:
            summary.append(f"  Measured nodes: {', '.join(sorted(measured_nodes))}")
        else:
            summary.append(f"No measured nodes provided.")
        
        if perturbed_nodes:
            summary.append(f"  Perturbed nodes: {', '.join(sorted(perturbed_nodes))}")
        else:
            summary.append(f"No perturbed nodes provided.")
        
        summary.append("")  # Empty line for separation
        
        non_obs = compression_info.get('removed_non_observable', set())
        non_ctrl = compression_info.get('removed_non_controllable', set())
        paths = compression_info.get('collapsed_paths', [])
        
        summary.append(f"Removed {len(non_obs)} non-observable nodes")
        if non_obs:
            summary.append(f"  Non-observable: {', '.join(sorted(non_obs))}")
        
        summary.append(f"Removed {len(non_ctrl)} non-controllable nodes")  
        if non_ctrl:
            summary.append(f"  Non-controllable: {', '.join(sorted(non_ctrl))}")
        
        summary.append(f"Collapsed {len(paths)} linear paths")
        if paths:
            for i, path in enumerate(paths, 1):
                # Show the complete path including the ultimate source
                if path:
                    # Get the ultimate source for proper display
                    src = self._ultimate_source(path[0]) if hasattr(self, '_ultimate_source') else path[0]
                    # Create the full path display: source -> path nodes -> target
                    full_path = [src] + path if src not in path else path
                    summary.append(f"  Path {i}: {' -> '.join(full_path)}")
        
        # Calculate total nodes removed correctly
        # For each path, count only the nodes that are actually removed
        total_removed_paths = 0
        for path in paths:
            if path and path[-1] in important_nodes:
                # Path ends at important node - only intermediate nodes removed
                total_removed_paths += len(path) - 1
            else:
                # All nodes in path removed
                total_removed_paths += len(path)
        
        total_removed = len(non_obs) + len(non_ctrl) + total_removed_paths
        summary.append(f"\nTotal nodes removed/collapsed: {total_removed}")
        summary.append(f"Final network size: {self.network.N} nodes")
        
        return "\n".join(summary)
    
    def visualize_compression(self, original_network, output_html="compression_visualization.html", 
                            interactive=False):
        """
        Visualize the compression results showing removed nodes and edges.
        
        Parameters:
        -----------
        original_network : BooleanNetwork
            The original network before compression
        output_html : str
            Output HTML file name
        interactive : bool
            If True, create interactive HTML visualization; if False, return matplotlib figure
            
        Returns:
        --------
        Network object if interactive=True, matplotlib figure if interactive=False, None on error
        """
        try:
            from ..BNMPy.vis import vis_compression_comparison
            
            compression_info = {
                'removed_non_observable': set(),
                'removed_non_controllable': set(), 
                'collapsed_paths': self.collapsed_paths,
                'removed_nodes': self.removed_nodes,
                'removed_edges': self.removed_edges,
                'measured_nodes': self.measured_nodes,
                'perturbed_nodes': self.perturbed_nodes
            }
            
            vis_compression_comparison(
                original_network, 
                self.network,
                compression_info,
                output_html,
                interactive
            )
            
        except ImportError:
            print("Visualization module not available. Please ensure BNMPy.vis is properly installed.")

    def _detect_output_nodes(self) -> Set[str]:
        """
        Automatically detect output nodes (nodes with no outgoing edges).
        
        Returns:
        --------
        Set[str]
            Set of output node names
        """
        output_nodes = set()
        
        for node_name in self.graph.nodes():
            # Check if node has no outgoing edges
            if self.graph.out_degree(node_name) == 0:
                output_nodes.add(node_name)
        
        return output_nodes

    def _get_rule(self, node_name: str) -> str:
        """Get the rule (right-hand side) for a given node."""
        if not hasattr(self.network, 'equations'):
            return ""
        
        for equation in self.network.equations:
            # Handle equations with extra whitespace around the equals sign
            eq_parts = equation.split('=', 1)
            if len(eq_parts) == 2 and eq_parts[0].strip() == node_name:
                return eq_parts[1].strip()
        return ""

    def _is_used_in_complex_rule(self, node: str) -> bool:
        """Return True if node appears in any rule as part of a complex expression (not a pure alias),
        or if the node's own rule is complex."""
        # Check if the node's own rule is complex
        own_rule = self._get_rule(node).strip()
        if own_rule and not self._alias_pat.fullmatch(own_rule):
            return True
            
        # Check if the node appears in other nodes' complex rules
        for eq in self.network.equations:
            lhs, rhs = eq.split('=', 1)
            rule = rhs.strip()
            # Node must appear in the rule, and the rule must not be a pure alias
            if node in rule and not self._alias_pat.fullmatch(rule):
                return True
        return False

    def _ultimate_source(self, var: str) -> str:
        """
        Follow alias chains (A = B = C ...) until a node is used in a complex rule or not a pure alias.
        """
        seen = set()
        while self._alias_pat.fullmatch(self._get_rule(var).strip()) and var not in seen:
            # Stop if this node is used in a complex rule
            if self._is_used_in_complex_rule(var):
                return var
            seen.add(var)
            var = self._get_rule(var).strip()
        return var

    def _collapse_alias_nodes(self) -> None:
        """
        Collapse alias nodes (nodes whose rule is a single identifier) by rewiring
        their children directly to their parent and removing the alias node.
        
        Important nodes (measured/perturbed) and nodes used in complex rules are never collapsed.
        """
        nodes_to_remove = set()
        
        # Important nodes that should never be collapsed
        important_nodes = self.measured_nodes | self.perturbed_nodes
        
        # Also preserve nodes used in complex rules
        complex_nodes = set()
        for node in self.graph.nodes():
            if self._is_used_in_complex_rule(node):
                complex_nodes.add(node)
        
        # Combined set of nodes that should never be collapsed
        protected_nodes = important_nodes | complex_nodes
        
        for node in list(self.graph.nodes()):
            if node in nodes_to_remove or node in protected_nodes:
                continue
                
            rule = self._get_rule(node).strip()
            
            # Check if this is an alias node (rule is a single identifier)
            if not self._alias_pat.fullmatch(rule):
                continue
                
            # Check if node has exactly one input and one output
            if self.graph.in_degree(node) != 1 or self.graph.out_degree(node) != 1:
                continue
                
            # Get parent and children
            parent = rule  # The node this alias points to
            children = list(self.graph.successors(node))
            
            if not children or parent not in self.graph.nodes():
                continue
                
            child = children[0]
            
            # Don't collapse if the child is also a protected node
            if child in protected_nodes:
                continue
            
            # Update child's rule to point directly to parent
            child_rule = self._get_rule(child)
            if child_rule:
                # Replace occurrences of alias node with parent
                pattern = r'\b' + re.escape(node) + r'\b'
                new_rule = re.sub(pattern, parent, child_rule)
                self._update_node_rule(child, new_rule)
            
            # Mark alias node for removal
            nodes_to_remove.add(node)
        
        # Remove all alias nodes
        if nodes_to_remove:
            self.remove_nodes(nodes_to_remove)
            # Remove equations for alias nodes
            if hasattr(self.network, 'equations'):
                self.network.equations = [eq for eq in self.network.equations 
                                        if eq.split('=')[0].strip() not in nodes_to_remove]
            # Remove nodes from nodeDict
            for node in nodes_to_remove:
                if node in self.network.nodeDict:
                    del self.network.nodeDict[node]
            
            # Rebuild graph after removal
            self.graph = self._build_graph()

def compress_model(network, measured_nodes: Set[str] = None, perturbed_nodes: Set[str] = None,
                  remove_non_observable: bool = True, remove_non_controllable: bool = True,
                  collapse_linear_paths: bool = True):
    """
    Convenience function to compress a model in one step.
    
    Parameters:
    -----------
    network : BooleanNetwork
        The Boolean network to compress
    measured_nodes : Set[str], optional
        Set of node names that are measured/observed
    perturbed_nodes : Set[str], optional
        Set of node names that are perturbed/controlled
    remove_non_observable : bool, default=True
        Whether to remove non-observable nodes
    remove_non_controllable : bool, default=True
        Whether to remove non-controllable nodes
    collapse_linear_paths : bool, default=True
        Whether to collapse linear paths
        
    Returns:
    --------
    Tuple[network, Dict]
        - Compressed network         
        - Compression information dictionary
    """
    # Check if it's a PBN
    if hasattr(network, 'cij'):
        raise ValueError("PBN compression is not currently supported. Please use Boolean Networks only.")
    
    # Make a deep copy to preserve the original network
    compressed_network = copy.deepcopy(network)
    
    compressor = ModelCompressor(compressed_network, measured_nodes, perturbed_nodes)
    
    compression_info = compressor.compress(
        remove_non_observable=remove_non_observable,
        remove_non_controllable=remove_non_controllable, 
        collapse_linear_paths=collapse_linear_paths
    )
     
    summary = compressor.get_compression_summary(compression_info)
    print(summary)
    
    return compressed_network, compression_info