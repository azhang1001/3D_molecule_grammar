import argparse

from loader import MoleculeDataset
from dataloader import DataLoaderAE
from util import NegativeEdge

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from tensorboardX import SummaryWriter

from fuseprop.chemutils import generate_conformers, calculate_geometric_features

criterion = nn.BCEWithLogitsLoss()

def train(args, model, device, loader, optimizer, epoch):
    model.train()
    for batch_idx, mol_graph in enumerate(loader):
        optimizer.zero_grad()
        # Extract features including geometric information
        x = mol_graph.x.to(device)
        edge_index = mol_graph.edge_index.to(device)
        edge_attr = mol_graph.edge_attr.to(device)
        geometric_features = mol_graph.get_geometric_features().to(device)  # Assuming modification

        output = model(x, edge_index, edge_attr, geometric_features)
        loss = F.binary_cross_entropy(output, mol_graph.edge_label.to(device))
        loss.backward()
        optimizer.step()
        # ... existing logging ...

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset, transform = NegativeEdge())

    print(dataset[0])

    loader = DataLoaderAE(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #set up model
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)   
    print(optimizer)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train(args, model, device, loader, optimizer, epoch)

    if not args.output_model_file == "":
        torch.save(model.state_dict(), args.output_model_file + ".pth")

if __name__ == "__main__":
    main()
