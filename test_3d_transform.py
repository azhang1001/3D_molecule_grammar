import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem import AllChem

from private.hypergraph import Hypergraph, mol_to_hg, hg_to_mol
from private.grammar import ProductionRule
from private.symbol import TSymbol, NTSymbol, BondSymbol

def draw_hypergraph_3d(hg, ax=None, title=None):
    """
    Draw a hypergraph in 3D with color-coded atom types, anchors, and non-terminals
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    if title:
        ax.set_title(title)
    
    # Colors for different atom types
    atom_colors = {
        'C': 'black',
        'H': 'lightgray',
        'O': 'red',
        'N': 'blue',
    }
    
    # Track atoms for bond drawing
    drawn_atoms = {}  # edge_name -> coords
    
    # Find anchor and non-terminal nodes
    anchor_nodes = set()
    for node in hg.nodes:
        try:
            if 'ext_id' in hg.node_attr(node):
                anchor_nodes.add(node)
        except:
            pass
    
    # Draw atoms (edges in hypergraph)
    for edge in hg.edges:
        # Skip edges without coordinates
        if edge not in hg.coordinates:
            continue
            
        coords = hg.get_coordinates(edge)
        edge_attr = hg.edge_attr(edge)
        
        # Get atom symbol
        if hasattr(edge_attr['symbol'], 'symbol'):
            symbol = edge_attr['symbol'].symbol
        else:
            # For non-terminal edges
            symbol = str(edge_attr['symbol'])
            
        is_terminal = edge_attr.get('terminal', True)
        
        # Check if this is an anchor atom (connected to anchor node)
        is_anchor = any(node in anchor_nodes for node in hg.nodes_in_edge(edge))
        
        # Determine the display style
        if not is_terminal:
            # Non-terminal node
            color = 'purple'
            size = 100
            label = 'NT'
        elif is_anchor and symbol == 'C':
            # Anchor carbon
            color = 'blue'
            size = 120
            label = 'C*'
        else:
            # Normal atom
            color = atom_colors.get(symbol, 'gray')
            size = 100
            label = symbol
            
        # Draw the atom
        ax.scatter(coords[0], coords[1], coords[2], color=color, s=size, edgecolor='black')
        ax.text(coords[0], coords[1], coords[2], label, fontsize=10)
        
        drawn_atoms[edge] = coords
    
    # Draw bonds
    for node in hg.nodes:
        # Find edges connected to this node (bond)
        connected_edges = []
        for edge in hg.edges:
            if node in hg.nodes_in_edge(edge) and edge in drawn_atoms:
                connected_edges.append(edge)
        
        # Draw bond between pairs of atoms
        if len(connected_edges) == 2:
            edge1, edge2 = connected_edges
            coords1 = drawn_atoms[edge1]
            coords2 = drawn_atoms[edge2]
            ax.plot([coords1[0], coords2[0]], 
                   [coords1[1], coords2[1]], 
                   [coords1[2], coords2[2]], 'k-', linewidth=2)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    return ax

def test_3d_rule_application():
    """
    Test the full 3D rule application pipeline from methane to ethane
    """
    
    # 1. Create a methane molecule with proper 3D tetrahedral structure
    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Convert to hypergraph
    methane = mol_to_hg(mol, kekulize=False, add_Hs=True)
    
    # 2. Identify a hydrogen to mark as non-terminal
    carbon_edge = None
    selected_h = None
    
    # Find carbon and one hydrogen
    for edge in methane.edges:
        if 'symbol' in methane.edge_attr(edge) and hasattr(methane.edge_attr(edge)['symbol'], 'symbol'):
            symbol = methane.edge_attr(edge)['symbol'].symbol
            if symbol == 'C':
                carbon_edge = edge
            elif symbol == 'H' and selected_h is None:
                selected_h = edge
    
    # Find the bond connecting them
    c_h_bond = None
    for node in methane.nodes_in_edge(carbon_edge):
        if node in methane.nodes_in_edge(selected_h):
            c_h_bond = node
            break
    
    # Get the bond symbol
    bond_symbol = methane.node_attr(c_h_bond)['symbol']
    
    # Mark the hydrogen as non-terminal
    methane.edge_attr(selected_h)['symbol'] = NTSymbol(
        degree=0, 
        is_aromatic=False,
        bond_symbol_list=[bond_symbol]
    )
    methane.edge_attr(selected_h)['terminal'] = False
    
    print(f"Starting methane structure:")
    print(f"  Carbon: {carbon_edge}")
    print(f"  Non-terminal hydrogen: {selected_h}")
    print(f"  C-H bond: {c_h_bond}")
    
    # 3. Create the production rule
    # LHS: Carbon connected to a non-terminal hydrogen
    lhs = Hypergraph()
    
    # Add the C-H bond
    lhs.add_node(c_h_bond, attr_dict={'symbol': bond_symbol})
    lhs.node_attr(c_h_bond)['ext_id'] = 0  # Mark as external/anchor
    
    # IMPORTANT: Add ONLY the non-terminal hydrogen first!
    # The ProductionRule.lhs_nt_symbol property takes the first edge's symbol
    lhs.add_edge([c_h_bond], attr_dict={
        'symbol': NTSymbol(degree=0, is_aromatic=False, bond_symbol_list=[bond_symbol]),
        'terminal': False
    }, edge_name="nt_h")
    
    # Debug info to verify symbols
    print("\nVerifying LHS symbols:")
    print(f"Methane hydrogen symbol: {methane.edge_attr(selected_h)['symbol']}")
    nt_symbol = lhs.edge_attr("nt_h")['symbol']
    print(f"LHS hydrogen symbol: {nt_symbol}")
    print(f"LHS first edge symbol: {lhs.edge_attr(list(lhs.edges)[0])['symbol']}")
    
    # Then add the carbon (optional, just for visualization)
    lhs.add_edge([c_h_bond], attr_dict={
        'symbol': TSymbol(degree=0, is_aromatic=False, symbol='C', 
                         num_explicit_Hs=0, formal_charge=0, chirality=0),
        'terminal': True
    }, edge_name="anchor_c")
    
    # Set coordinates for LHS atoms
    lhs.set_coordinates("anchor_c", methane.get_coordinates(carbon_edge))
    if selected_h in methane.coordinates:
        lhs.set_coordinates("nt_h", methane.get_coordinates(selected_h))
    
    # RHS: Carbon connected to CH3 group
    rhs = Hypergraph()
    
    # Add the anchor C-H bond
    rhs.add_node(c_h_bond, attr_dict={'symbol': bond_symbol})
    rhs.node_attr(c_h_bond)['ext_id'] = 0  # Mark as external/anchor
    
    # Add C-C bond
    cc_bond = "c_c_bond"
    rhs.add_node(cc_bond, attr_dict={'symbol': BondSymbol(is_aromatic=False, bond_type=1, stereo=0)})
    
    # Add the anchor carbon (connected to the original molecule)
    rhs.add_edge([c_h_bond, cc_bond], attr_dict={
        'symbol': TSymbol(degree=0, is_aromatic=False, symbol='C', 
                         num_explicit_Hs=0, formal_charge=0, chirality=0),
        'terminal': True
    }, edge_name="anchor_c")
    
    # Create CH bonds for the methyl group
    ch_bonds = []
    for i in range(3):
        ch_bond = f"ch_bond_{i}"
        rhs.add_node(ch_bond, attr_dict={'symbol': BondSymbol(is_aromatic=False, bond_type=1, stereo=0)})
        ch_bonds.append(ch_bond)
    
    # Add the new carbon (center of the methyl group)
    rhs.add_edge([cc_bond] + ch_bonds, attr_dict={
        'symbol': TSymbol(degree=0, is_aromatic=False, symbol='C', 
                         num_explicit_Hs=0, formal_charge=0, chirality=0),
        'terminal': True
    }, edge_name="methyl_c")
    
    # Add the hydrogens for the methyl group
    for i, ch_bond in enumerate(ch_bonds):
        rhs.add_edge([ch_bond], attr_dict={
            'symbol': TSymbol(degree=0, is_aromatic=False, symbol='H', 
                             num_explicit_Hs=0, formal_charge=0, chirality=0),
            'terminal': True
        }, edge_name=f"methyl_h{i}")
    
    # Set 3D coordinates for RHS atoms
    carbon_pos = methane.get_coordinates(carbon_edge)
    rhs.set_coordinates("anchor_c", carbon_pos)
    
    # Calculate a position for the new carbon (1.5Å away from the original carbon)
    # Use the direction from carbon to the non-terminal hydrogen
    h_pos = methane.get_coordinates(selected_h)
    direction = h_pos - carbon_pos
    direction = direction / np.linalg.norm(direction)
    
    # Position the new carbon
    new_carbon_pos = carbon_pos + 1.5 * direction
    rhs.set_coordinates("methyl_c", new_carbon_pos)
    
    # Calculate tetrahedral positions for the methyl hydrogens
    # Create vectors that form a tetrahedron with the C-C bond
    cc_vector = new_carbon_pos - carbon_pos
    cc_vector = cc_vector / np.linalg.norm(cc_vector)
    
    # Find orthogonal vectors to create a tetrahedral arrangement
    if abs(cc_vector[0]) < abs(cc_vector[1]):
        perp = np.array([0, cc_vector[2], -cc_vector[1]])
    else:
        perp = np.array([cc_vector[2], 0, -cc_vector[0]])
    perp = perp / np.linalg.norm(perp)
    
    # Create a third perpendicular vector
    third = np.cross(cc_vector, perp)
    
    # Create tetrahedral directions (excluding the direction back to carbon)
    tetrahedral_dirs = [
        -cc_vector + perp + third,
        -cc_vector - perp - third,
        -cc_vector + perp - third
    ]
    
    # Position the methyl hydrogens
    bond_length = 1.1  # Standard C-H bond length
    for i, direction in enumerate(tetrahedral_dirs):
        direction = direction / np.linalg.norm(direction)
        h_pos = new_carbon_pos + bond_length * direction
        rhs.set_coordinates(f"methyl_h{i}", h_pos)
    
    # Create the production rule
    rule = ProductionRule(lhs, rhs)
    
    # Verify the production rule's LHS symbol matches the methane hydrogen
    print("\nVerifying rule LHS symbol:")
    print(f"Rule LHS symbol: {rule.lhs_nt_symbol}")
    print(f"Methane hydrogen symbol: {methane.edge_attr(selected_h)['symbol']}")
    print(f"Symbols match: {rule.lhs_nt_symbol == methane.edge_attr(selected_h)['symbol']}")
    
    # 4. Apply the rule to the methane
    print("\nApplying production rule...")
    result, mapping, success = rule.applied_to(methane, selected_h, return_success=True)
    
    if success:
        print("Rule application successful!")
        
        # Count atoms to verify we have ethane (C2H6)
        carbon_count = 0
        hydrogen_count = 0
        
        for edge in result.edges:
            symbol = result.edge_attr(edge)['symbol']
            if hasattr(symbol, 'symbol'):
                if symbol.symbol == 'C':
                    carbon_count += 1
                elif symbol.symbol == 'H':
                    hydrogen_count += 1
        
        print(f"Result structure: C{carbon_count}H{hydrogen_count}")
        
        # Verify through SMILES
        try:
            result_mol = hg_to_mol(result)
            smiles = Chem.MolToSmiles(result_mol)
            print(f"Result SMILES: {smiles}")
            print(f"Is ethane: {smiles == 'CC'}")
        except Exception as e:
            print(f"Error converting to molecule: {e}")
    else:
        print("Rule application failed")
    
    # 5. Visualize all the structures
    fig = plt.figure(figsize=(18, 5))
    
    # Draw original methane
    ax1 = fig.add_subplot(141, projection='3d')
    draw_hypergraph_3d(methane, ax=ax1, title="Methane (Original)")
    
    # Draw LHS
    ax2 = fig.add_subplot(142, projection='3d')
    draw_hypergraph_3d(lhs, ax=ax2, title="LHS: C-H")
    
    # Draw RHS
    ax3 = fig.add_subplot(143, projection='3d')
    draw_hypergraph_3d(rhs, ax=ax3, title="RHS: C-CH3")
    
    # Draw result
    ax4 = fig.add_subplot(144, projection='3d')
    if success:
        draw_hypergraph_3d(result, ax=ax4, title="Ethane (Result)")
    else:
        ax4.set_title("Rule Application Failed")
    
    plt.tight_layout()
    plt.savefig("3d_rule_application.png", dpi=300)
    plt.show()
    
    return methane, rule, result

if __name__ == "__main__":
    methane, rule, result = test_3d_rule_application()
    print("Test completed!")