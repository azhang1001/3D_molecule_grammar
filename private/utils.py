from copy import deepcopy
from typing import List
import numpy as np
import os
import logging
import shutil
from scipy.spatial.transform import Rotation


def _node_match(node1, node2):
    # if the nodes are hyperedges, `atom_attr` determines the match
    if node1['bipartite'] == 'edge' and node2['bipartite'] == 'edge':
        return node1["attr_dict"]['symbol'] == node2["attr_dict"]['symbol']
    elif node1['bipartite'] == 'node' and node2['bipartite'] == 'node':
        # bond_symbol
        return node1['attr_dict']['symbol'] == node2['attr_dict']['symbol']
    else:
        return False

def _easy_node_match(node1, node2):
    # if the nodes are hyperedges, `atom_attr` determines the match
    if node1['bipartite'] == 'edge' and node2['bipartite'] == 'edge':
        return node1["attr_dict"].get('symbol', None) == node2["attr_dict"].get('symbol', None)
    elif node1['bipartite'] == 'node' and node2['bipartite'] == 'node':
        # bond_symbol
        return node1['attr_dict'].get('ext_id', -1) == node2['attr_dict'].get('ext_id', -1)\
            and node1['attr_dict']['symbol'] == node2['attr_dict']['symbol']
    else:
        return False


def _node_match_prod_rule(node1, node2, ignore_order=False):
    # if the nodes are hyperedges, `atom_attr` determines the match
    if node1['bipartite'] == 'edge' and node2['bipartite'] == 'edge':
        return node1["attr_dict"]['symbol'] == node2["attr_dict"]['symbol']
    elif node1['bipartite'] == 'node' and node2['bipartite'] == 'node':
        # ext_id, order4hrg, bond_symbol
        if ignore_order:
            return node1['attr_dict']['symbol'] == node2['attr_dict']['symbol']
        else:
            return node1['attr_dict']['symbol'] == node2['attr_dict']['symbol']\
                and node1['attr_dict'].get('ext_id', -1) == node2['attr_dict'].get('ext_id', -1)
    else:
        return False


def _edge_match(edge1, edge2, ignore_order=False):
    #return True
    if ignore_order:
        return True
    else:
        return edge1["order"] == edge2["order"]

def masked_softmax(logit, mask):
    ''' compute a probability distribution from logit

    Parameters
    ----------
    logit : array-like, length D
        each element indicates how each dimension is likely to be chosen
        (the larger, the more likely)
    mask : array-like, length D
        each element is either 0 or 1.
        if 0, the dimension is ignored
        when computing the probability distribution.

    Returns
    -------
    prob_dist : array, length D
        probability distribution computed from logit.
        if `mask[d] = 0`, `prob_dist[d] = 0`.
    '''
    if logit.shape != mask.shape:
        raise ValueError('logit and mask must have the same shape')
    c = np.max(logit)
    exp_logit = np.exp(logit - c) * mask
    sum_exp_logit = exp_logit @ mask
    return exp_logit / sum_exp_logit


def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def align_substructure(fixed_coords, moving_coords, connection_points):
    """
    Align a moving substructure to a fixed structure at connection points.
    
    Parameters
    ----------
    fixed_coords : dict
        Mapping of node names to fixed coordinates
    moving_coords : dict
        Mapping of node names to coordinates that will be transformed
    connection_points : list of tuples
        List of (fixed_node, moving_node) pairs that should be aligned
        
    Returns
    -------
    dict
        Transformed coordinates for the moving substructure
    """
    if not connection_points:
        return moving_coords
    
    # For a single connection point, we translate to match positions
    if len(connection_points) == 1:
        fixed_node, moving_node = connection_points[0]
        translation = fixed_coords[fixed_node] - moving_coords[moving_node]
        
        # Apply translation to all moving coordinates
        transformed_coords = {node: coords + translation 
                             for node, coords in moving_coords.items()}
        return transformed_coords
    
    # For multiple connection points, we need rotation + translation
    elif len(connection_points) >= 2:
        # Get the connection points
        fixed_points = np.array([fixed_coords[node] for node, _ in connection_points])
        moving_points = np.array([moving_coords[node] for _, node in connection_points])
        
        # Center both point sets
        fixed_center = np.mean(fixed_points, axis=0)
        moving_center = np.mean(moving_points, axis=0)
        
        # Center the points
        fixed_centered = fixed_points - fixed_center
        moving_centered = moving_points - moving_center
        
        # Compute optimal rotation using Kabsch algorithm
        H = moving_centered.T @ fixed_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (not reflection)
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = Vt.T @ U.T
        
        # Calculate translation
        t = fixed_center - R @ moving_center
        
        # Apply transformation to all moving coordinates
        transformed_coords = {}
        for node, coords in moving_coords.items():
            transformed_coords[node] = R @ coords + t
            
        return transformed_coords
    
    return moving_coords

def transfer_coordinates_from_rule(result, original_graph, rule, mapping=None):
    """
    Transfer 3D coordinates from a rule's RHS to the result hypergraph.
    
    Parameters
    ----------
    result : Hypergraph
        The result hypergraph after rule application
    original_graph : Hypergraph
        The original hypergraph before rule application
    rule : ProductionRule
        The production rule that was applied
    mapping : dict, optional
        Mapping from rule edges to result edges
        
    Returns
    -------
    result : Hypergraph
        The result hypergraph with proper 3D coordinates
    """
    # First copy existing coordinates from original graph
    for edge in result.edges:
        if edge in original_graph.edges and edge in original_graph.coordinates:
            result.set_coordinates(edge, original_graph.get_coordinates(edge))
    
    # Then add new coordinates from the rule's RHS
    # This requires figuring out the mapping from RHS edge names to result edge names
    rhs_to_result = {}
    
    # If we have an explicit mapping from the rule application, use it
    if mapping:
        for rhs_edge, result_edge in mapping.items():
            if isinstance(rhs_edge, str) and isinstance(result_edge, str):
                rhs_to_result[rhs_edge] = result_edge
    
    # Otherwise, try to match by edge name
    for edge in result.edges:
        if edge not in original_graph.edges:
            # This is a new edge added by the rule
            if edge in rule.rhs.edges:
                rhs_to_result[edge] = edge
    
    # Transfer coordinates
    for rhs_edge, result_edge in rhs_to_result.items():
        if rhs_edge in rule.rhs.coordinates:
            coords = rule.rhs.coordinates[rhs_edge]
            result.set_coordinates(result_edge, coords)
    
    return result

def transfer_coordinates_after_rule_application(result_hg, original_hg, rule_rhs, edge_map_rhs, replaced_edge):
    """Transfer 3D coordinates after a production rule has been applied."""
    
    # First, copy existing coordinates from original hypergraph
    for edge in result_hg.edges:
        if edge in original_hg.edges and edge in original_hg.coordinates:
            result_hg.set_coordinates(edge, original_hg.get_coordinates(edge))
    
    # Debug: Show current state of coordinates
    print(f"After copying original coordinates:")
    print(f"  Edges with coordinates: {[e for e in result_hg.edges if e in result_hg.coordinates]}")
    print(f"  Missing coordinates for: {[e for e in result_hg.edges if e not in result_hg.coordinates]}")
    
    # Find edges with and without coordinates
    edges_with_coords = set(edge for edge in result_hg.edges if edge in result_hg.coordinates)
    edges_without_coords = set(result_hg.edges) - edges_with_coords
    
    # If all edges have coordinates already, we're done
    if not edges_without_coords:
        return result_hg
    
    # Find anchor points between original and RHS
    # These are edges with the same name in both graphs
    anchor_edges = []
    for rhs_edge in rule_rhs.edges:
        if rhs_edge in original_hg.edges and rhs_edge in original_hg.coordinates and rhs_edge in rule_rhs.coordinates:
            anchor_edges.append(rhs_edge)
    
    # If no direct anchor edges, find edges that got transferred to result
    if not anchor_edges:
        anchor_edges_original = []
        anchor_edges_rhs = []
        for node in original_hg.nodes:
            # Only use nodes that were preserved in the result
            if node in result_hg.nodes:
                # Find edges connected to this node in original graph
                for edge in original_hg.edges:
                    if node in original_hg.nodes_in_edge(edge) and edge in original_hg.coordinates:
                        # Find corresponding RHS edge (should be connected to same node)
                        for rhs_edge in rule_rhs.edges:
                            if node in rule_rhs.nodes_in_edge(rhs_edge) and rhs_edge in rule_rhs.coordinates:
                                anchor_edges_original.append(edge)
                                anchor_edges_rhs.append(rhs_edge)
                                break
    else:
        # Use exact matches as anchors
        anchor_edges_original = anchor_edges
        anchor_edges_rhs = anchor_edges
    
    # Debug: Show anchor edges
    print(f"Anchor edges:")
    print(f"  Original: {anchor_edges_original}")
    print(f"  RHS: {anchor_edges_rhs}")
    
    # Backup approach - if we have a carbon atom in both, use that
    if not anchor_edges_original:
        for edge in original_hg.edges:
            if (edge in original_hg.coordinates and 
                hasattr(original_hg.edge_attr(edge)['symbol'], 'symbol') and 
                original_hg.edge_attr(edge)['symbol'].symbol == 'C'):
                for rhs_edge in rule_rhs.edges:
                    if (rhs_edge in rule_rhs.coordinates and 
                        hasattr(rule_rhs.edge_attr(rhs_edge)['symbol'], 'symbol') and 
                        rule_rhs.edge_attr(rhs_edge)['symbol'].symbol == 'C'):
                        anchor_edges_original.append(edge)
                        anchor_edges_rhs.append(rhs_edge)
                        break
                break
    
    # If we still have no anchor points, we need a different approach
    if not anchor_edges_original:
        print("WARNING: No anchor points found between original and RHS")
        
        # Map RHS edges to result edges based on edge_map_rhs
        for rhs_edge, result_edge in edge_map_rhs.items():
            if rhs_edge in rule_rhs.coordinates and result_edge in edges_without_coords:
                # Direct transfer without transformation
                result_hg.set_coordinates(result_edge, rule_rhs.get_coordinates(rhs_edge))
                
        # Edge names in RHS might directly match edge names in result
        for edge in edges_without_coords:
            if edge in rule_rhs.coordinates:
                result_hg.set_coordinates(edge, rule_rhs.get_coordinates(edge))
        
        return result_hg
    
    # Calculate transformation between coordinate systems
    anchor_points_original = np.array([original_hg.get_coordinates(edge) for edge in anchor_edges_original])
    anchor_points_rhs = np.array([rule_rhs.get_coordinates(edge) for edge in anchor_edges_rhs])
    
    # If we have just one anchor point, we can only do translation
    if len(anchor_edges_original) == 1:
        # Calculate translation vector
        translation = anchor_points_original[0] - anchor_points_rhs[0]
        rotation = np.eye(3)  # Identity rotation
    else:
        # Calculate optimal rotation and translation
        rotation, translation = calculate_optimal_transform(anchor_points_rhs, anchor_points_original)
    
    print(f"Calculated transformation:")
    print(f"  Rotation: {rotation}")
    print(f"  Translation: {translation}")
    
    # Apply transformation to all edges in RHS that don't have coordinates in result
    edges_added = 0
    
    # First try using edge_map_rhs
    for rhs_edge, result_edge in edge_map_rhs.items():
        if result_edge in edges_without_coords and rhs_edge in rule_rhs.coordinates:
            # Get RHS coordinates
            rhs_coords = rule_rhs.get_coordinates(rhs_edge)
            
            # Apply transformation
            transformed_coords = np.dot(rotation, rhs_coords) + translation
            
            # Set coordinates in result
            result_hg.set_coordinates(result_edge, transformed_coords)
            edges_added += 1
    
    # Try direct matching by edge name if any still missing
    for edge in edges_without_coords:
        if edge in result_hg.coordinates:
            continue  # Already added
            
        if edge in rule_rhs.coordinates:
            # Direct match by name
            rhs_coords = rule_rhs.get_coordinates(edge)
            transformed_coords = np.dot(rotation, rhs_coords) + translation
            result_hg.set_coordinates(edge, transformed_coords)
            edges_added += 1
    
    # Debug: Show updated coordinates
    print(f"After transformation:")
    print(f"  Added coordinates for {edges_added} edges")
    print(f"  Edges with coordinates: {[e for e in result_hg.edges if e in result_hg.coordinates]}")
    print(f"  Missing coordinates for: {[e for e in result_hg.edges if e not in result_hg.coordinates]}")
    
    # If we still have missing coordinates, try to guess based on the transformation
    if edges_without_coords - set(result_hg.coordinates.keys()):
        print("WARNING: Some edges still missing coordinates, trying to guess based on RHS")
        for edge in edges_without_coords:
            if edge not in result_hg.coordinates:
                # Try to find a matching edge in RHS by symbol or pattern
                for rhs_edge in rule_rhs.edges:
                    if (rhs_edge in rule_rhs.coordinates and
                        hasattr(rule_rhs.edge_attr(rhs_edge)['symbol'], 'symbol') and
                        hasattr(result_hg.edge_attr(edge)['symbol'], 'symbol') and
                        rule_rhs.edge_attr(rhs_edge)['symbol'].symbol == result_hg.edge_attr(edge)['symbol'].symbol):
                        
                        # Found a matching edge by symbol, transform coordinates
                        rhs_coords = rule_rhs.get_coordinates(rhs_edge)
                        transformed_coords = np.dot(rotation, rhs_coords) + translation
                        result_hg.set_coordinates(edge, transformed_coords)
                        break
    
    # Final verification
    edges_with_coords_final = set(edge for edge in result_hg.edges if edge in result_hg.coordinates)
    edges_without_coords_final = set(result_hg.edges) - edges_with_coords_final
    
    if edges_without_coords_final:
        print(f"WARNING: {len(edges_without_coords_final)} edges still missing coordinates")
        print(f"  Missing: {edges_without_coords_final}")
    else:
        print("SUCCESS: All edges have coordinates")
        
    return result_hg

def calculate_optimal_transform(points_a, points_b):
    """
    Calculate the optimal rigid transformation (rotation + translation)
    to align points_a with points_b using the Kabsch algorithm.
    
    Parameters
    ----------
    points_a : numpy.ndarray
        Array of shape (n, 3) containing the first set of points
    points_b : numpy.ndarray
        Array of shape (n, 3) containing the second set of points
        
    Returns
    -------
    rotation : numpy.ndarray
        3x3 rotation matrix
    translation : numpy.ndarray
        Translation vector
    """
    # Ensure we're working with numpy arrays
    points_a = np.array(points_a)
    points_b = np.array(points_b)
    
    # Center the points
    centroid_a = np.mean(points_a, axis=0)
    centroid_b = np.mean(points_b, axis=0)
    
    centered_a = points_a - centroid_a
    centered_b = points_b - centroid_b
    
    # Calculate the cross-covariance matrix
    H = np.dot(centered_a.T, centered_b)
    
    # Singular value decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Calculate rotation matrix
    rotation = np.dot(Vt.T, U.T)
    
    # Ensure we have a right-handed coordinate system
    if np.linalg.det(rotation) < 0:
        Vt[-1,:] *= -1
        rotation = np.dot(Vt.T, U.T)
    
    # Calculate translation
    translation = centroid_b - np.dot(rotation, centroid_a)
    
    return rotation, translation

def align_3d_structure(target_coords, template_coords, anchor_indices):
    """
    Align a 3D molecular structure to a template structure based on anchor atoms.
    
    Parameters
    ----------
    target_coords : dict
        Dictionary mapping atom indices to coordinates that need to be aligned
    template_coords : dict
        Dictionary mapping atom indices to coordinates that serve as the template
    anchor_indices : list
        List of atom indices that serve as anchors for alignment
        
    Returns
    -------
    aligned_coords : dict
        Dictionary mapping atom indices to aligned coordinates
    """
    # Extract anchor coordinates
    anchor_points_target = np.array([target_coords[idx] for idx in anchor_indices])
    anchor_points_template = np.array([template_coords[idx] for idx in anchor_indices])
    
    # Calculate optimal transformation
    rotation, translation = calculate_optimal_transform(anchor_points_target, anchor_points_template)
    
    # Apply transformation to all target coordinates
    aligned_coords = {}
    for idx, coords in target_coords.items():
        aligned_coords[idx] = np.dot(rotation, coords) + translation
    
    return aligned_coords
