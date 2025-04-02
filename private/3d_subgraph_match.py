import networkx as nx
import numpy as np
from itertools import permutations

def is_subgraph_match(G1, G2, epsilon=1e-3):
    GM = nx.algorithms.isomorphism.GraphMatcher(
        G2, G1, 
        node_match=lambda n1, n2: n1['atom_type'] == n2['atom_type']
    )
    
    for mapping in GM.subgraph_isomorphisms_iter():
        inv_mapping = {v: k for k, v in mapping.items()}
        mapped_positions = {n: np.array(G2.nodes[inv_mapping[n]]['pos']) for n in G1.nodes}
        
        for u, v in G1.edges:
            d1 = np.linalg.norm(np.array(G1.nodes[u]['pos']) - np.array(G1.nodes[v]['pos']))
            d2 = np.linalg.norm(mapped_positions[u] - mapped_positions[v])
            if abs(d1 - d2) > epsilon:
                break  
        else:
            return True  
    
    return False  

