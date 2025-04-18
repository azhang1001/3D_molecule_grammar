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

def transfer_coordinates_after_rule_application(result_hg, original_hg, rhs, edge_map_rhs=None, replaced_edge=None):
    """
    Transfer coordinates from original hypergraph and RHS to the result
    after applying a production rule
    
    Parameters
    ----------
    result_hg : Hypergraph
        The resulting hypergraph after rule application
    original_hg : Hypergraph
        The original hypergraph before rule application
    rhs : Hypergraph
        The right-hand side of the applied rule
    edge_map_rhs : dict
        Mapping from RHS edge names to result edge names
    replaced_edge : str
        Name of the edge that was replaced in the original hypergraph
    
    Returns
    -------
    result_hg : Hypergraph
        The result hypergraph with updated coordinates
    """
    # Copy existing coordinates from original hypergraph
    for edge in original_hg.edges:
        if edge != replaced_edge and edge in result_hg.edges and edge in original_hg.coordinates:
            result_hg.set_coordinates(edge, original_hg.coordinates[edge])
    
    # Find anchor edges to compute transformation
    # These are edges that exist in both the original hypergraph and the result
    anchor_edges_orig = []
    anchor_edges_rhs = []
    
    # Find edges in RHS that connect to anchor nodes (ext_id)
    for edge in rhs.edges:
        is_anchor = False
        for node in rhs.nodes_in_edge(edge):
            if "ext_id" in rhs.node_attr(node):
                is_anchor = True
                break
        
        if is_anchor and edge in rhs.coordinates and edge_map_rhs.get(edge) in result_hg.edges:
            mapped_edge = edge_map_rhs[edge]
            
            # Find corresponding edge in original hypergraph
            for orig_edge in original_hg.edges:
                if orig_edge in result_hg.edges and orig_edge == mapped_edge:
                    # This is a preserved edge, use it as an anchor
                    if orig_edge in original_hg.coordinates:
                        anchor_edges_orig.append(orig_edge)
                        anchor_edges_rhs.append(edge)
        
    # Debug info
    print("Anchor edges:")
    print(f"  Original: {anchor_edges_orig}")
    print(f"  RHS: {anchor_edges_rhs}")
    
    # Calculate transformation to align RHS with original structure
    if anchor_edges_orig and anchor_edges_rhs:
        # Collect coordinates for anchor points
        orig_coords = np.array([original_hg.coordinates[e] for e in anchor_edges_orig])
        rhs_coords = np.array([rhs.coordinates[e] for e in anchor_edges_rhs])
        
        # There are two ways to fix this:
        
        # Option 1: Create connection_points for align_substructure
        connection_points = list(zip(range(len(anchor_edges_orig)), range(len(anchor_edges_rhs))))
        
        # Option 2: Use calculate_optimal_transform which only needs two point sets
        R, t = calculate_optimal_transform(rhs_coords, orig_coords)
        
        print("Calculated transformation:")
        print(f"  Rotation: {R}")
        print(f"  Translation: {t}")
        
        # Apply transformation to all RHS coordinates and copy to result
        for rhs_edge, result_edge in edge_map_rhs.items():
            if rhs_edge in rhs.coordinates and result_edge not in result_hg.coordinates:
                # Transform coordinates
                coords = rhs.coordinates[rhs_edge]
                new_coords = np.dot(R, coords) + t
                
                # Set in result
                result_hg.set_coordinates(result_edge, new_coords)
    
    # Check if there are any edges still missing coordinates
    missing_coords = [e for e in result_hg.edges if e not in result_hg.coordinates]
    if missing_coords:
        print(f"Missing coordinates for: {missing_coords}")
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
