import torch
import numpy as np
import pandas as pd
import networkx as nx
import pyflagser
import statistics
from collections import Counter
from torch_geometric.utils import degree, to_networkx
import torch.nn as nn
import torch.nn.functional as F

def process_thresholds(lst, N):
    """
    Calculates thresholds based on value frequency and distribution.
    """
    if N < 2:
        raise ValueError("N must be at least 2")
    
    if isinstance(lst, torch.Tensor):
        lst = lst.tolist()
        
    count = Counter(lst)
    min_val, max_val = min(lst), max(lst)
    
    count.pop(min_val, None)
    count.pop(max_val, None)
    
    top_values = sorted(count.items(), key=lambda x: x[1], reverse=True)[:N - 1]
    thresholds = [min_val] + [value for value, _ in top_values] + [max_val]
    return sorted(thresholds)

def stat(acc_list, metric):
    mean = statistics.mean(acc_list)
    stdev = statistics.stdev(acc_list)
    print('Final', metric, f'using 5 fold CV: {mean:.4f} \u00B1 {stdev:.4f}%')

def print_stat(train_acc, test_acc):
    argmax = np.argmax(train_acc)
    best_result = test_acc[argmax]
    train_ac = np.max(train_acc)
    test_ac = np.max(test_acc)
    return test_ac, best_result

def apply_Zscore(MP_tensor):
    mean = MP_tensor.mean(dim=(0, 2, 3), keepdim=True)   
    std  = MP_tensor.std(dim=(0, 2, 3), keepdim=True)    
    MP_tensor_z = (MP_tensor - mean) / (std + 1e-8)  
    return MP_tensor_z

def calculate_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    eps = 1e-9
    probs_clamped = torch.clamp(probs, eps, 1 - eps)
    entropy = -torch.sum(probs_clamped * torch.log2(probs_clamped), dim=-1)
    entropy = torch.nan_to_num(entropy, nan=0.0)
    return torch.mean(entropy)

def get_degree_centrality(data):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    deg = degree(edge_index[0], num_nodes=num_nodes)
    deg_centrality = deg / (num_nodes - 1) if num_nodes > 1 else deg
    return deg_centrality

def compute_degree_centrality(graph):
    deg = nx.degree_centrality(graph)
    nodes = list(graph.nodes())
    deg_vals = [deg[n] for n in nodes]
    return torch.tensor(deg_vals, dtype=torch.float32).view(-1, 1)

def compute_closeness_centrality(graph):
    clo = nx.closeness_centrality(graph)
    nodes = list(graph.nodes())
    clo_vals = [clo[n] for n in nodes]
    return torch.tensor(clo_vals, dtype=torch.float32).view(-1, 1)

def get_atomic_weight(data):
    if data.x is not None and data.x.shape[1] > 0:
        Atomic_weight = [data.x[i][0] for i in range(len(data.x))]
        return torch.tensor(Atomic_weight)
    return torch.zeros(data.num_nodes)

def compute_hks(graph, t_values):
    if graph.number_of_nodes() == 0:
        return torch.tensor([])
        
    L = nx.laplacian_matrix(graph).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)
    
    if isinstance(t_values, torch.Tensor):
        t_val = t_values.item()
    else:
        t_val = t_values
        
    heat_kernel = np.dot(eigvecs, np.dot(np.diag(np.exp(-t_val * eigvals)), eigvecs.T))
    hks = []
    for i, node in enumerate(graph.nodes()):
        hks.append(heat_kernel[i, i])
    return torch.tensor(hks)

def get_thresh_hks(dataset, number_threshold, t_value):
    graph_list = []
    label = []
    for graph_id in range(len(dataset)):
        graph = to_networkx(dataset[graph_id], to_undirected=True)
        if graph.number_of_nodes() > 0:
            hks_values = compute_hks(graph, t_value)
        else:
            hks_values = torch.tensor([0.0])
        graph_list.append(hks_values)
        label.append(dataset[graph_id].y)
    thresh = torch.cat(graph_list, dim=0)
    thresh = process_thresholds(thresh, number_threshold)
    return graph_list, torch.tensor(thresh), label

def get_thresh_atom(dataset, number_threshold):
    thresh = []
    graph_list = []
    for graph_id in range(len(dataset)):
        atomic_values = get_atomic_weight(dataset[graph_id])
        graph_list.append(atomic_values)
    thresh = torch.cat(graph_list, dim=0)
    thresh = process_thresholds(thresh, number_threshold)
    return graph_list, torch.tensor(thresh)

def get_thresh(dataset, number_threshold):
    thresh = []
    graph_list = []
    for graph_id in range(len(dataset)):
        degree_centrality_values = get_degree_centrality(dataset[graph_id])
        graph_list.append(degree_centrality_values)
    thresh = torch.cat(graph_list, dim=0)
    thresh = process_thresholds(thresh, number_threshold)
    return graph_list, torch.tensor(thresh)

def compute_topological_features(dataset, number_threshold, t_value, filtration_function):
    graph_list = []
    label = []
    for graph_id in range(len(dataset)):
        graph = to_networkx(dataset[graph_id], to_undirected=True)
        if filtration_function == 'hks':
            topo_value = compute_hks(graph, t_value)
        elif filtration_function == 'deg':
            topo_value = compute_degree_centrality(graph)
        elif filtration_function == 'close':
            topo_value = compute_closeness_centrality(graph)
        else:
            print("Filtration function is not valid")
            topo_value = torch.tensor([])
        graph_list.append(topo_value)
        label.append(dataset[graph_id].y)
        
    thresh = torch.cat(graph_list, dim=0)
    thresh = process_thresholds(thresh, number_threshold)
    return graph_list, torch.tensor(thresh), label

def get_Topo_Fe(graph, feature, threshold):
    betti0_row = []
    betti1_row = []
    nodes_row = []
    edges_row = []
    edge_index = graph.edge_index
    feature = feature.view(-1)
    threshold = threshold.view(-1)
    
    for p in range(threshold.size(0)):
        idx1 = torch.where(feature <= threshold[p])[0]
        n_active = torch.tensor(list(set(idx1.tolist())), dtype=torch.long)
        if n_active.numel() == 0:
            betti0_row.append(0); betti1_row.append(0)
            nodes_row.append(0); edges_row.append(0)
        else:
            active_set = set(n_active.tolist())
            G = nx.Graph()
            G.add_nodes_from(active_set)
            u = edge_index[0].tolist()
            v = edge_index[1].tolist()
            for uu, vv in zip(u, v):
                if uu in active_set and vv in active_set:
                    G.add_edge(int(uu), int(vv))
            Adj = nx.to_numpy_array(G, nodelist=sorted(active_set))
            my_flag = pyflagser.flagser_unweighted(
                Adj, min_dimension=0, max_dimension=2,
                directed=False, coeff=2, approximation=None
            )
            x = my_flag["betti"]
            betti0_row.append(int(x[0]))
            betti1_row.append(int(x[1]) if len(x) > 1 else 0)
            nodes_row.append(len(active_set))
            edges_row.append(G.number_of_edges())
            
    return torch.cat((torch.tensor(betti0_row, dtype=torch.float),
            torch.tensor(betti1_row, dtype=torch.float),
            torch.tensor(nodes_row, dtype=torch.float),
            torch.tensor(edges_row, dtype=torch.float)), dim=0)

class ConcatFusion(nn.Module):
    def __init__(self, gnn_dim, topo_dim, emb_dim):
        super().__init__()
        self.in_dim = gnn_dim + topo_dim
        self.out_dim = emb_dim
        self.fusion_layer = nn.Linear(self.in_dim, self.out_dim)
    def forward(self, g_emb, t_emb): return F.relu(self.fusion_layer(torch.cat([g_emb, t_emb], dim=-1)))
