import torch
import os
import warnings
from torch_geometric.datasets import TUDataset
from ogb.graphproppred import PygGraphPropPredDataset
import torch_geometric.transforms as T

warnings.filterwarnings("ignore", ".*index_select.*")

DATA_ROOT = "./data"

def load_dataset(dataset_name):
    dataset = TUDataset(
        root=os.path.join(DATA_ROOT, dataset),
        name=dataset_name,
        transform=T.ToUndirected()
    )
    return dataset
    
def load_ogb_dataset(dataset_name):
    dataset = PygGraphPropPredDataset(
        name=dataset_name,
        root=DATA_ROOT 
    )
    return dataset
    
    

