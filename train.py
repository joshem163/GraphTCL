import os
import sys
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Batch, Dataset, InMemoryDataset
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch.cuda.amp import GradScaler, autocast
import argparse
import warnings
import logging
import numpy as np
import networkx as nx
from collections import Counter
from torch_geometric.utils import degree, to_networkx
from modules import *
from model import *
from data_loader import *

def contrastive_loss(z1, z2, temperature=0.1, eps=1e-5):
    z1 = F.normalize(z1 + eps, dim=-1)
    z2 = F.normalize(z2 + eps, dim=-1)
    sim_matrix = torch.matmul(torch.cat([z1, z2], dim=0), torch.cat([z1, z2], dim=0).t()) / temperature
    mask = torch.eye(2 * z1.size(0), device=z1.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, torch.finfo(sim_matrix.dtype).min)
    labels = torch.cat([torch.arange(z1.size(0), 2*z1.size(0), device=z1.device), torch.arange(0, z1.size(0), device=z1.device)])
    return F.cross_entropy(sim_matrix, labels)

def collate_graph_topo(batch):
    graphs, topos, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(topos), torch.stack(labels)

def train(model, loader, optimizer, device, scaler, alpha=0.1):
    model.train()
    total_correct = 0
    total_examples = 0
    
    for batch_graphs, batch_topo_data, batch_labels in loader:
        if batch_graphs.num_graphs == 0: continue
        
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)

        if isinstance(batch_topo_data, tuple):
            batch_topo = (batch_topo_data[0].to(device), batch_topo_data[1].to(device))
        else:
            batch_topo = batch_topo_data.to(device)

        if batch_labels.dim() > 1: 
            batch_labels = batch_labels.squeeze()
            
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits, _, _, g_emb, t_emb = model(
                batch_graphs.x, 
                batch_graphs.edge_index, 
                batch_graphs.batch, 
                topo_feats=batch_topo
            )
            loss = F.cross_entropy(logits, batch_labels) + alpha * contrastive_loss(g_emb, t_emb)

        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=-1)
        total_correct += (preds == batch_labels).sum().item()
        total_examples += batch_graphs.num_graphs
        
    return 0.0, 0.0, total_correct / total_examples if total_examples > 0 else 0


def evaluate(model, loader, device):
    model.eval()
    total_correct = 0
    total_examples = 0
    
    with torch.no_grad():
        for batch_graphs, batch_topo_data, batch_labels in loader:
            batch_graphs = batch_graphs.to(device)
            batch_labels = batch_labels.to(device)
            
            if isinstance(batch_topo_data, tuple):
                batch_topo = (batch_topo_data[0].to(device), batch_topo_data[1].to(device))
            else:
                batch_topo = batch_topo_data.to(device)

            if batch_labels.dim() > 1: 
                batch_labels = batch_labels.squeeze()
                
            logits, _, _, _, _ = model(
                batch_graphs.x, 
                batch_graphs.edge_index, 
                batch_graphs.batch, 
                topo_feats=batch_topo
            )
            
            preds = logits.argmax(dim=-1)
            total_correct += (preds == batch_labels).sum().item()
            total_examples += batch_graphs.num_graphs
            
    return total_correct / total_examples if total_examples > 0 else 0, 0.0

def run(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset = TUDataset(root=f"./data/{args.dataset}", name=args.dataset, transform=T.ToUndirected())
    if dataset[0].x is None or dataset[0].x.shape[1] == 0:
        new_list = []
        max_degree = 0

        for data in dataset:
            if data.num_nodes > 0:
                d = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
                max_degree = max(max_degree, d.max().item() if d.numel() > 0 else 0)

        for data in dataset:
            if data.num_nodes == 0:
                 data.x = torch.zeros((0, max_degree + 1), dtype=torch.float)
            else:
                d = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
                d[d > max_degree] = max_degree
                data.x = F.one_hot(d, num_classes=max_degree + 1).float()

            new_list.append(data)
        dataset = MyDataset(new_list)

    cache_path = f"./topo_cache/topo_cache_{args.dataset}.pt"
    if not os.path.exists(cache_path):

       
        dataset = TUDataset(root='./data', name=args.dataset)

        raw_signal_list, thresholds, _ = compute_topological_features(
            dataset=dataset, 
            number_threshold=10, 
            t_value=0.1, 
            filtration_function=args.filtration
        )

        graph_features = []
        print(f" -> Extracting Betti numbers using {args.filtration} filtration...")
        
        for i in tqdm(range(len(dataset)), desc="Topo Extraction"):
            topo_fe = get_Topo_Fe(dataset[i], raw_signal_list[i], thresholds)
            graph_features.append(topo_fe.float())

        topo_tensor = torch.stack(graph_features)
        
        labels = dataset.y.squeeze()

        save_dict = {
            'y': labels,
            'topo_tensor': topo_tensor
        }
        
        torch.save(save_dict, cache_path)

        del dataset, raw_signal_list, graph_features, topo_tensor
        gc.collect()
        return
    cache = torch.load(cache_path)
    topo_tensor = cache['topo_tensor']
    topo_tensor = (topo_tensor - topo_tensor.mean(dim=0)) / (topo_tensor.std(dim=0) + 1e-6)

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    try: y_all = dataset.y.numpy()
    except: y_all = [d.y.item() for d in dataset]
    splits = list(skf.split(torch.zeros(len(dataset)), y_all))
    train_idx, test_idx = splits[args.fold_id - 1]

    data_train = [dataset[i] for i in train_idx]; data_test = [dataset[i] for i in test_idx]
    topo_train = topo_tensor[train_idx]; topo_test = topo_tensor[test_idx]
    y_train = torch.tensor([y_all[i] for i in train_idx]).long(); y_test = torch.tensor([y_all[i] for i in test_idx]).long()

    train_loader = DataLoader(GraphWithTopoDataset(data_train, topo_train, y_train), batch_size=args.batch_size, shuffle=True, collate_fn=collate_graph_topo, num_workers=args.num_workers)
    test_loader = DataLoader(GraphWithTopoDataset(data_test, topo_test, y_test), batch_size=args.batch_size, shuffle=False, collate_fn=collate_graph_topo, num_workers=args.num_workers)

    if args.model_type.lower() == 'gin':
        gnn = GINEncoder(dataset[0].x.shape[1], args.num_layer, args.emb_dim, args.dropout)
    else:
        gnn = GCNEncoder(dataset[0].x.shape[1], args.num_layer, args.emb_dim, args.dropout)
    topo = MLPEncoder(topo_tensor.shape[1], args.emb_dim)
    fuse = ConcatFusion(gnn.out_dim, topo.out_dim, args.emb_dim)
    head = nn.Linear(fuse.out_dim, num_tasks)
    model = HybridGraphTopoModel(gnn, topo, fuse, head, args.emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    torch.compile()
    scaler = GradScaler()

    best_test_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        _, _, _ = train(model, train_loader, optimizer, device, scaler, alpha=args.alpha)
        test_acc, _ = evaluate(model, test_loader, device)
        if test_acc > best_test_acc: best_test_acc = test_acc

    print(f"[Fold {args.fold_id}] Best Acc: {best_test_acc:.4f}", flush=True)

<<<<<<< HEAD


=======
>>>>>>> 6256c8aed25bb5415e3ad0a0105e009a4212321d
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fold_id', type=int, required=True)
    parser.add_argument('--model_type', type=str, default='gin')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--device_arg', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--filtration', type=str, choices=['hks', 'deg', 'close'], default='hks')
    args = parser.parse_args()
    load_dataset(args.dataset)
    try:
        run(args)
    except Exception as e:
        print(f"RUNTIME FAILURE: {e}")
        traceback.print_exc()
        sys.exit(1)
