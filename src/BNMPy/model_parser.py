import re
# import libsbml
import os
import boolean
from collections import defaultdict

def simplify_expression(expression):
    """
    Simplifies the Boolean expression by removing redundant parentheses using boolean.py package.
    
    :param expression: The Boolean expression as a string.
    :return: The simplified expression.
    """
    algebra = boolean.BooleanAlgebra()
    algebra.parse(expression)
    simplified_expression = str(algebra.parse(expression).simplify())
    # Replace '~' with '!'
    simplified_expression = simplified_expression.replace('~', '!')

    # add space around the operators |, &, =, (, )
    simplified_expression = re.sub(r'([&|=\(\)])', r' \1 ', simplified_expression)
    # remove more than one spaces
    simplified_expression = re.sub(r'\s+', ' ', simplified_expression)
    simplified_expression = simplified_expression.strip()

    return simplified_expression

def parse_expression(expression: str):
    """
    Parses a Boolean expression to separate activators and inhibitors.
    """
    activators = []
    inhibitors = []

    # deal with complex inhibitors
    complex_blocks = re.findall(r'!\(([^)]+)\)', expression)
    for block in complex_blocks:
        # every gene inside the negated block is an inhibitor
        genes = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', block)
        inhibitors.extend(genes)

    # strip them out so they don’t confuse the simple scan
    expression = re.sub(r'!\([^)]*\)', '', expression)

    # scan the remaining text
    token_re = re.compile(r'!?\b[A-Za-z_][A-Za-z0-9_]*\b')
    for m in token_re.finditer(expression):
        token = m.group(0)
        if token.startswith('!'):
            inhibitors.append(token[1:])          # drop the '!'
        else:
            activators.append(token)

    # tidy up duplicates
    activators = list(set(activators))
    inhibitors = list(set(inhibitors))
    return activators, inhibitors

# Parse equations into dicts: {gene: rule}
def parse_equations(equations):
    eq_dict = {}
    for line in equations:
        if '=' in line:
            target, rule = line.split('=', 1)
            rule = simplify_expression(rule.strip())
            eq_dict[target.strip()] = rule
    return eq_dict

def get_upstream(rule_str):
    from BNMPy.BMatrix import get_upstream_genes
    eq_str = f"X = {rule_str}"
    return get_upstream_genes([eq_str])[0].split()

def find_direct_targets(target_nodes, rule_dict):
    direct_targets = set()
    for gene, rule in rule_dict.items():
        upstreams = set(get_upstream(rule))
        if upstreams & target_nodes:
            direct_targets.add(gene)
    return direct_targets

def check_inhibitor_wins_rule(expression):
    """
    Ensures that the "Inhibitor Wins" rule is respected within a single expression.
    """
    components = re.split(r'\|', expression)
    
    for component in components:
        component = component.strip()
        if '&' in component:
            activators, inhibitors = parse_expression(component)
            if activators and inhibitors:
                continue
        if "!" not in component:
            continue
        activators, inhibitors = parse_expression(component)
        if activators and inhibitors:
            return False
    
    return True

def merge_networks(BNs, method="OR", prob=0.9, descriptive=False):
    """
    Merge multiple Boolean-network models using one of four strategies.

    Parameters
    ----------
    BNs : list[BooleanNetwork]
        List of Boolean-network objects
    method : {"OR", "AND", "Inhibitor Wins", "PBN"}
        Merge strategy.
    prob : float, default 0.9
        Only used for PBN.  Probability given to equations from the first model
    descriptive : bool, default False
        If True, prints a human-readable merge log.

    Returns
    -------
    str
        A newline-separated network definition string suitable for writing to file.
        • deterministic modes:   gene = Boolean-expression
        • PBN mode:              gene = expr1, p1\n
                                 gene = expr2, p2 ...  (one line per rule)
    """
    if method not in ["OR", "AND", "Inhibitor Wins", "PBN"]:
        raise ValueError("Invalid method. Use 'OR', 'AND', 'Inhibitor Wins', or 'PBN'")

    network_dicts = []
    for BN in BNs:
        network_dicts.append(parse_equations(BN.equations))

    algebra        = boolean.BooleanAlgebra()
    merged_network = {}
    descriptions   = defaultdict(list)
    warning_gene   = defaultdict(list)
    overlap_genes  = set()
    individual_gene_counts = [len(net) for net in network_dicts]

    if method == "PBN":
        gene_occ = defaultdict(int)
        for net in network_dicts:
            for g in net:
                gene_occ[g] += 1
        if len(BNs) > 1:
            prob_other = (1 - prob) / (len(BNs) - 1)
        else:
            prob_other = 0.0

    for idx, network in enumerate(network_dicts, start=1):

        for gene, expression in network.items():

            # PBN MODE
            if method == "PBN":
                if gene_occ[gene] > 1:
                    overlap_genes.add(gene)

                # probability assignment
                rule_prob = 1.0 if gene_occ[gene] == 1 else (prob if idx == 1 else prob_other)
                rule_str  = f"{expression}, {round(rule_prob, 3)}"

                # avoid duplicate rules
                existing_rules = [r.split(',')[0].strip() for r in merged_network.get(gene, [])]
                if expression in existing_rules:
                    descriptions[gene].append((f"Model {idx}", f"duplicate {expression}"))
                    continue

                merged_network.setdefault(gene, []).append(rule_str)
                descriptions[gene].append((f"Model {idx}", rule_str))
                descriptions[gene].append(("Merged", merged_network[gene]))
                continue  # go to next gene

            # DETERMINISTIC MODES
            if gene in merged_network:
                overlap_genes.add(gene)

                # identical rule
                if algebra.parse(merged_network[gene]) == algebra.parse(expression):
                    if descriptive:
                        descriptions[gene].append((f"Model {idx}", expression))
                        descriptions[gene].append(("Merged", merged_network[gene]))
                    continue

                # merge according to method
                if method == "OR":
                    merged_expression = f"({merged_network[gene]})|({expression})"
                elif method == "AND":
                    merged_expression = f"({merged_network[gene]})&({expression})"
                elif method == "Inhibitor Wins":
                    activators, inhibitors = parse_expression(expression)
                    existing_act, existing_inh = parse_expression(merged_network[gene])

                    act_all = list(set(existing_act + activators))
                    inh_all = list(set(existing_inh + inhibitors))

                    # resolve genes acting both ways
                    overlap = set(act_all) & set(inh_all)
                    if overlap:
                        warning_gene[gene].extend(overlap)
                        act_all = [a for a in act_all if a not in overlap]

                    act_expr = '|'.join(filter(None, act_all)) if act_all else ""
                    inh_expr = '|'.join(filter(None, inh_all)) if inh_all else ""

                    merged_expression = (f"({act_expr})&!({inh_expr})"
                                         if inh_expr else act_expr)
                else:
                    merged_expression = expression  # fallback
                merged_network[gene] = simplify_expression(merged_expression)
                descriptions[gene].append((f"Model {idx}", expression))
                descriptions[gene].append(("Merged", merged_network[gene]))

            # first time this gene appears
            else:
                if method == "PBN":
                    # for unique genes we *already* handled prob=1 rule above,
                    # but this branch never executes in PBN.
                    pass
                merged_network[gene] = expression
                descriptions[gene].append((f"Model {idx}", expression))

    if descriptive:
        print(f"Merging Method: {method}")
        print(f"Total Genes in Merged Network: {len(merged_network)}")
        print("Number of Genes in Each Individual Model:")
        for i, cnt in enumerate(individual_gene_counts, 1):
            print(f"  Model {i}: {cnt} genes")

        print(f"Overlapping Genes: {len(overlap_genes)}")
        if overlap_genes:
            print("Overlapping Genes List: " + ', '.join(sorted(overlap_genes)))
            for gene in sorted(overlap_genes):
                if method != "PBN":
                    print(f"\nGene: {gene}")
                if warning_gene.get(gene):
                    print(f"Warning: possible conflicts for {warning_gene[gene]}, keeping only inhibitor.")
                for lbl, expr in descriptions[gene]:
                    if method == "PBN":
                        pass
                    else:
                        print(f"  {lbl} Function: {expr}")
        else:
            print("No overlapping genes found.")

    # return the merged network string
    merged_lines = []
    if method == "PBN":
        # each rule is already stored as "expr, p"  (string)
        for gene, rules in merged_network.items():
            for rule in rules:             # rule = "expr, prob"
                merged_lines.append(f"{gene} = {rule}")
    else:
        for gene, expr in merged_network.items():
            merged_lines.append(f"{gene} = {expr}")
    # sort alphabetically
    merged_lines.sort()
    merged_network_string = "\n".join(merged_lines)
    return merged_network_string

def BN2PBN(bn_string, prob=0.5):
    """
    Expand the boolean network to a PBN by adding a self-loop as alternative function
    prob: probability of the equations from the original BN

    Returns:
        pbn_string: string of the PBN
        nodes_to_optimize: list of nodes excludes input nodes
    """
    # Parse equations from BN
    bn_equations = {}
    for line in bn_string.strip().split('\n'):
        if '=' in line:
            target, rule = line.split('=', 1)
            bn_equations[target.strip()] = rule.strip()
    
    # Expand rules
    pbn_equations = []
    nodes_to_optimize = []

    for target in bn_equations.keys():
        if bn_equations[target] == target:
            # If its already a self-loop (e.g., a input node)
            pbn_equations.append(f"{target} = {bn_equations[target]}, 1")
        else:
            # Add the original rule with a prob
            pbn_equations.append(f"{target} = {bn_equations[target]}, {prob}")
            # Add the alternative rule
            pbn_equations.append(f"{target} = {target}, {round(1-prob,2)}")
            nodes_to_optimize.append(target)
    
    # remove equations with prob = 0
    pbn_equations = [eq for eq in pbn_equations if eq.split(',')[1] != ' 0']
    pbn_string = '\n'.join(pbn_equations)
    return pbn_string, nodes_to_optimize


def extend_networks(original_network, new_network, nodes_to_extend, prob=0.5, descriptive=True):
    """
    Extend the original network by adding the nodes_to_extend (and their rules) from the new network.
    This will return a PBN where:
    - rules for the nodes_to_extend are added to the original network with a probability of prob.
    - rules for the nodes_to_extend from the original network will have a probability of 1-prob.
    - in case the new rules for nodes_to_extend include new nodes, they will also be added with a probability of 1.
    - The direct targets of nodes_to_extend are also extended with both rules and probabilities.
    - The rest of the rules are kept the same with a probability of 1.
    """
    orig_dict = parse_equations(original_network.equations)
    new_dict = parse_equations(new_network.equations)

    added_genes = set()
    rules = []  # list of (gene, rule, prob)

    # Keep track of which genes have had their rules modified (compared to original)
    affected_genes = set()
    modified_rules = {}

    # 1. Add nodes_to_extend with both new and original rules (if present in both)
    for node in nodes_to_extend:
        if node in orig_dict and node in new_dict:
            rules.append((node, new_dict[node], prob))
            rules.append((node, orig_dict[node], round(1-prob, 6)))
            added_genes.add(node)
            if orig_dict[node] != new_dict[node]:
                affected_genes.add(node)
                modified_rules[node] = {
                    "original": orig_dict[node],
                    "new": new_dict[node] + f", {prob}"
                }
        elif node in orig_dict:
            rules.append((node, orig_dict[node], 1.0))
            added_genes.add(node)
        elif node in new_dict:
            rules.append((node, new_dict[node], 1.0))
            added_genes.add(node)
            # If not in original, it's a new node, so mark as affected
            affected_genes.add(node)
            modified_rules[node] = {
                "original": None,
                "new": new_dict[node] + ", 1"
            }
        else:
            raise ValueError(f"Node {node} is not in the original or new network")
        # Add any new upstream genes from the new rule (with prob=1)
        if node in new_dict:
            for up_gene in get_upstream(new_dict[node]):
                if up_gene not in orig_dict and up_gene in new_dict and up_gene not in added_genes:
                    rules.append((up_gene, new_dict[up_gene], 1.0))
                    added_genes.add(up_gene)
                    # New node, so mark as affected
                    affected_genes.add(up_gene)
                    modified_rules[up_gene] = {
                        "original": None,
                        "new": new_dict[up_gene] + ", 1"
                    }

    # 2. Add direct targets of nodes_to_extend
    direct_targets = find_direct_targets(set(nodes_to_extend), orig_dict)
    direct_targets |= find_direct_targets(set(nodes_to_extend), new_dict)
    for node in direct_targets:
        if node in added_genes:
            continue
        if node in orig_dict and node in new_dict:
            rules.append((node, new_dict[node], prob))
            rules.append((node, orig_dict[node], round(1-prob, 6)))
            added_genes.add(node)
            if orig_dict[node] != new_dict[node]:
                affected_genes.add(node)
                modified_rules[node] = {
                    "original": orig_dict[node],
                    "new": new_dict[node] + f", {prob}"
                }
        elif node in orig_dict:
            rules.append((node, orig_dict[node], 1.0))
            added_genes.add(node)
        elif node in new_dict:
            rules.append((node, new_dict[node], 1.0))
            added_genes.add(node)
            # If not in original, it's a new node, so mark as affected
            affected_genes.add(node)
            modified_rules[node] = {
                "original": None,
                "new": new_dict[node] + ", 1"
            }
        else:
            raise ValueError(f"Node {node} is not in the original or new network")

    # 3. Add all other genes from the original network (not already added)
    for gene in orig_dict:
        if gene not in added_genes:
            rules.append((gene, orig_dict[gene], 1.0))
            added_genes.add(gene)
    # 4. Add any remaining new genes (from new network) that were not in original, if not already added
    for gene in new_dict:
        if gene not in added_genes:
            rules.append((gene, new_dict[gene], 1.0))
            added_genes.add(gene)
            # If not in original, it's a new node, so mark as affected
            affected_genes.add(gene)
            modified_rules[gene] = {
                "original": None,
                "new": new_dict[gene] + ", 1"
            }
    # 5. Format as PBN string, remove any with prob=0, and sort
    pbn_lines = []
    for gene, rule, p in rules:
        if p == 0:
            continue
        pbn_lines.append(f"{gene} = {rule}, {p}")
    pbn_lines.sort()
    pbn_string = '\n'.join(pbn_lines)

    if descriptive:
        print(f"Nodes affected: {sorted(affected_genes)}")
        for gene in sorted(affected_genes):
            print(f"\nGene: {gene}")
            orig_rule = modified_rules[gene].get("original")
            new_rule = modified_rules[gene].get("new")
            if orig_rule is not None:
                print(f"  Original rule: {orig_rule}")
            else:
                print(f"  Not present in original network")
            if new_rule is not None:
                print(f"  Added rule: {new_rule}")
            else:
                print(f"  Not present in new network")
    return pbn_string