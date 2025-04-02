import torch
import os, random, gc
import pickle

from rdkit import Chem
from torch.utils.data import Dataset
from fuseprop.chemutils import random_subgraph, extract_subgraph, enum_root
from fuseprop.mol_graph import MolGraph

class MoleculeDataset(Dataset):

    def __init__(self, root, dataset, transform=None):
        self.data = self.load_dataset(root, dataset)
        self.transform = transform

    def load_dataset(self, root, dataset):
        """
        Load molecules and generate MolGraph instances with 3D features.
        """
        with open(os.path.join(root, dataset + ".txt"), 'r') as f:
            smiles_list = f.read().splitlines()
        
        mol_graphs = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol_graph = MolGraph(mol)
                mol_graphs.append(mol_graph)
        return mol_graphs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mol_graph = self.data[idx]
        if self.transform:
            mol_graph = self.transform(mol_graph)
        return mol_graph


class ReconstructDataset(Dataset):

    def __init__(self, data, avocab, batch_size):
        self.batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.avocab = avocab
    
    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        subgraphs = []
        init_smiles = []
        for smiles in self.batches[idx]:
            mol = Chem.MolFromSmiles(smiles)
            selected_atoms = random_subgraph(mol, ratio=0.5)
            sub_smiles, root_atoms = extract_subgraph(smiles, selected_atoms)
            subgraph = MolGraph(smiles, selected_atoms, root_atoms, shuffle_roots=False)
            subgraphs.append(subgraph)
            init_smiles.append(sub_smiles)
        return MolGraph.tensorize(subgraphs), self.batches[idx], init_smiles


class SubgraphDataset(Dataset):

    def __init__(self, data, avocab, batch_size, num_decode):
        data = [x for smiles in data for x in enum_root(smiles, num_decode)]
        self.batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.avocab = avocab
    
    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


class DataFolder(object):

    def __init__(self, data_folder, batch_size, shuffle=True):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                batches = pickle.load(f)

            if self.shuffle: random.shuffle(batches) #shuffle data before batch
            for batch in batches:
                yield batch

            del batches
            gc.collect()

