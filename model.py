import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder

class GINEncoder(nn.Module):
    def __init__(self, in_dim, num_layers, emb_dim, dropout):
        super(GINEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.out_dim = emb_dim

        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(in_dim, 2 * emb_dim),
            nn.BatchNorm1d(2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )))
        self.bns.append(nn.BatchNorm1d(emb_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))))
            self.bns.append(nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = x.float()
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return x

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, num_layers, emb_dim, dropout):
        super(GCNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        hidden_dim = emb_dim
        self.out_dim = emb_dim
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.fc_out = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = x.float()
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        return self.fc_out(x)

class MLPEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(MLPEncoder, self).__init__()
        self.out_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, emb_dim), nn.ReLU(), nn.BatchNorm1d(emb_dim), nn.Linear(emb_dim, emb_dim))
    def forward(self, x): return self.mlp(x)

class HybridGraphTopoModel(nn.Module):
    def __init__(self, gnn_encoder, topo_encoder, fusion_module, classifier_head, emb_dim):
        super(HybridGraphTopoModel, self).__init__()
        self.gnn_encoder = gnn_encoder
        self.topo_encoder = topo_encoder
        self.fusion_module = fusion_module
        self.classifier_head = classifier_head
        hidden_dim = self.fusion_module.out_dim
        proj_dim = emb_dim // 2
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim), nn.BatchNorm1d(proj_dim))
    def forward(self, x, edge_index, batch, topo_feats, edge_attr=None, labels=None):
        g_emb = self.gnn_encoder(x, edge_index, batch, edge_attr=edge_attr)
        t_emb = self.topo_encoder(topo_feats)
        fused = self.fusion_module(g_emb, t_emb)
        logits = self.classifier_head(fused)
        proj = F.normalize(self.projection_head(fused), dim=-1)
        return logits, proj, fused, g_emb, t_emb

class HybridGraphTopoModelOGB(nn.Module):
    def __init__(self, gnn_encoder, topo_encoder, fusion_module, classifier_head, emb_dim):
        super(HybridGraphTopoModel, self).__init__()
        self.atom_encoder = AtomEncoder(emb_dim)
        self.gnn_encoder = gnn_encoder
        self.topo_encoder = topo_encoder
        self.fusion_module = fusion_module
        self.classifier_head = classifier_head
        hidden_dim = self.fusion_module.out_dim
        proj_dim = emb_dim // 2
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim)
        )

    def forward(self, x, edge_index, batch, topo_feats, edge_attr=None, labels=None):
        x_emb = self.atom_encoder(x.long())
        g_emb = self.gnn_encoder(x_emb, edge_index, batch, edge_attr=edge_attr)
        t_emb = self.topo_encoder(topo_feats)
        fused = self.fusion_module(g_emb, t_emb)
        logits = self.classifier_head(fused)
        proj = F.normalize(self.projection_head(fused), dim=-1)
        return logits, proj, fused, g_emb, t_emb

class GraphWithTopoDataset(Dataset):
    def __init__(self, graphs, topo_feats, labels):
        self.graphs = graphs
        self.topo_feats = topo_feats
        self.labels = labels
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx], self.topo_feats[idx], self.labels[idx]

class MyDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.')
        self.data, self.slices = self.collate(data_list)

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
