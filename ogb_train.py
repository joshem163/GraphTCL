import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GCNConv, GINEConv, global_mean_pool, global_add_pool
from torch_geometric.data import Batch, Dataset, InMemoryDataset
import torch_geometric.transforms as T
from torch.cuda.amp import GradScaler, autocast
import sys
import argparse
import warnings
import logging
import numpy as np
import networkx as nx
from collections import Counter
from torch_geometric.utils import degree, to_networkx
from tqdm import tqdm
import random
from ogb.graphproppred.mol_encoder import AtomEncoder
from libauc.losses import MultiLabelAUCMLoss
from libauc.optimizers import PESG
from ogb.graphproppred.mol_encoder import BondEncoder
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from data_loader import *
from model import *
from modules import *

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

def train(model, loader, optimizer, device, scaler, cls_criterion, alpha=0.1, task_type="classification"):
    model.train()
    total_loss = 0
    total_entropy = 0
    total_examples = 0

    for i, (batch_graphs, batch_topo_data, batch_labels) in enumerate(loader):
        if batch_graphs.num_graphs == 0: continue
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)
        if isinstance(batch_topo_data, tuple):
            batch_topo = (batch_topo_data[0].to(device), batch_topo_data[1].to(device))
        else:
            batch_topo = batch_topo_data.to(device)

        if batch_labels.ndim == 3 and batch_labels.size(1) == 1:
            batch_labels = batch_labels.squeeze(1)

        is_labeled = ~torch.isnan(batch_labels)
        if not is_labeled.any(): continue
        batch_labels_clean = torch.nan_to_num(batch_labels, nan=0.0)

        optimizer.zero_grad()
        logits, proj, _, g_emb, t_emb = model(
                batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch,
                topo_feats=batch_topo,
                edge_attr=batch_graphs.edge_attr,
                labels=batch_labels_clean
            )
        cls_loss = cls_criterion(logits, batch_labels_clean.float())
        con_loss = contrastive_loss(g_emb.float(), t_emb.float())
        loss = cls_loss + alpha * con_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            batch_entropy = calculate_entropy(logits)
            total_entropy += batch_entropy.item() * batch_graphs.num_graphs

        total_loss += loss.item() * batch_graphs.num_graphs
        total_examples += batch_graphs.num_graphs

    if total_examples == 0: return 0.0, 0.0
    return total_loss / total_examples, total_entropy / total_examples

@torch.no_grad()
def evaluate(model, loader, device, evaluator, task_type="classification"):
    model.eval()
    y_true, y_pred = [], []
    total_entropy = 0
    total_examples = 0

    for batch_graphs, batch_topo_data, batch_labels in loader:
        if batch_graphs.num_graphs == 0: continue
        num_graphs_in_batch = batch_graphs.num_graphs
        total_examples += num_graphs_in_batch
        batch_graphs = batch_graphs.to(device)
        if isinstance(batch_topo_data, tuple):
            batch_topo = (batch_topo_data[0].to(device), batch_topo_data[1].to(device))
        else:
            batch_topo = batch_topo_data.to(device)

        logits, _, _, _, _ = model(
            batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch,
            topo_feats=batch_topo,
            edge_attr=batch_graphs.edge_attr,
            labels=None
        )
        batch_entropy = calculate_entropy(logits)
        total_entropy += batch_entropy.item() * num_graphs_in_batch
        y_true.append(batch_labels.view(logits.shape).cpu())
        y_pred.append(logits.cpu())

    if not y_true: return {evaluator.eval_metric: 0.0, 'entropy': 0.0}
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    try:
        eval_results = evaluator.eval(input_dict)
    except Exception as e:
        print(f"Eval warning: {e}")
        eval_results = {evaluator.eval_metric: 0.0}
    avg_entropy = total_entropy / total_examples if total_examples > 0 else 0.0
    eval_results['entropy'] = avg_entropy
    return eval_results

def run(args):
    set_seed(args.fold_id + 42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PygGraphPropPredDataset(name=args.dataset, root="data")
    evaluator = Evaluator(args.dataset)

    if dataset[0].x is None or dataset[0].x.shape[1] == 0:
        new_list = []
        max_degree = 0
        for data in dataset:
            d = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, d.max().item() if d.numel() > 0 else 0)
        for data in dataset:
            d = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
            d[d > max_degree] = max_degree
            data.x = F.one_hot(d, num_classes=max_degree + 1).float()
            new_list.append(data)
        dataset_processed = MyDataset(new_list)
    else:
        new_list = []
        for data in dataset:
            data.x = data.x.float()
            new_list.append(data)
        dataset_processed = MyDataset(new_list)

    cache_path = f"/topo_cache/topo_cache_{args.dataset}.pt"
  
    if not os.path.exists(cache_path):
        dataset = PygGraphPropPredDataset(name=args.dataset, root='/data')
        raw_signal_list, thresholds, _ = compute_topological_features(
            dataset=dataset, 
            number_threshold=10,        
            t_value=0.1,                
            filtration_function=args.filtration
        )
    
        graph_features = []
        
        for i in tqdm(range(len(dataset)), desc="Topo Extraction"):
            topo_fe = get_Topo_Fe(dataset[i], raw_signal_list[i], thresholds)
            graph_features.append(topo_fe.float())
    
        topo_tensor = torch.stack(graph_features)
        
        save_dict = {
            'y': dataset.y,
            'topo_tensor': topo_tensor
        }
        
        torch.save(save_dict, cache_path)
    
        del dataset, raw_signal_list, graph_features, topo_tensor
        gc.collect()
        return
      
    cache = torch.load(cache_path)
    topo_tensor = cache['topo_tensor']
    

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    train_dataset = [dataset_processed[i] for i in train_idx]
    valid_dataset = [dataset_processed[i] for i in valid_idx]
    test_dataset = [dataset_processed[i] for i in test_idx]

    topo_train = topo_tensor[train_idx]
    topo_valid = topo_tensor[valid_idx]
    topo_test = topo_tensor[test_idx]

    y_train = torch.cat([dataset_processed[i].y for i in train_idx], dim=0)
    y_valid = torch.cat([dataset_processed[i].y for i in valid_idx], dim=0)
    y_test = torch.cat([dataset_processed[i].y for i in test_idx], dim=0)

    train_loader = DataLoader(GraphWithTopoDataset(train_dataset, topo_train, y_train),
                              batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_graph_topo, num_workers=args.num_workers)
    valid_loader = DataLoader(GraphWithTopoDataset(valid_dataset, topo_valid, y_valid),
                              batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_graph_topo, num_workers=args.num_workers)
    test_loader = DataLoader(GraphWithTopoDataset(test_dataset, topo_test, y_test),
                             batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_graph_topo, num_workers=args.num_workers)

    if args.model_type.lower() == 'gin':
        gnn = GINEncoder(args.emb_dim, args.num_layer, args.emb_dim, args.dropout)
    else:
        gnn = GCNEncoder(args.emb_dim, args.num_layer, args.emb_dim, args.dropout)

    topo = MLPEncoder(topo_tensor.shape[1], args.emb_dim)
    fuse = ConcatFusion(gnn.out_dim, topo.out_dim, args.emb_dim)
    num_tasks = dataset.num_tasks
    head = nn.Linear(fuse.out_dim, num_tasks)
    model = HybridGraphTopoModelOGB(gnn, topo, fuse, head, args.emb_dim).to(device)
    cls_criterion = MultiLabelAUCMLoss(num_labels=num_tasks)
    torch.compile(model)
    optimizer = PESG(model.parameters(), loss_fn=cls_criterion, lr=args.lr, margin=1.0)
    scaler = GradScaler('cuda')

    best_valid_score = 0.0
    best_test_score = 0.0
    is_maximizing = True

    for epoch in range(1, args.epochs + 1):
        loss, _ = train(model, train_loader, optimizer, device, scaler, cls_criterion, alpha=args.alpha)
        valid_dict = evaluate(model, valid_loader, device, evaluator)
        test_dict = evaluate(model, test_loader, device, evaluator)

        valid_score = valid_dict[evaluator.eval_metric]
        test_score = test_dict[evaluator.eval_metric]

        improved = False
        if is_maximizing:
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                best_test_score = test_score
                improved = True
        else:
            if valid_score < best_valid_score:
                best_valid_score = valid_score
                best_test_score = test_score
                improved = True

    print(f"[Seed {args.fold_id}] Final Best Test {evaluator.eval_metric.upper()}: {best_test_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fold_id', type=int, required=True)
    parser.add_argument('--model_type', type=str, default='gin')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_layer', type=int, default=5)
    parser.add_argument('--emb_dim', type=int, default=448)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--filtration', type=str, choices=['hks', 'deg', 'close'], default='deg')
    args = parser.parse_args()
    load_ogb_dataset(args.dataset)
    try:
        run(args)
    except Exception as e:
        print(f"RUNTIME FAILURE: {e}")
        traceback.print_exc()
        sys.exit(1)
