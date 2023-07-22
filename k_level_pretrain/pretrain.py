import os
import json
import pickle
import networkx as nx
import numpy as np
from KGNN import KGNN
from tqdm import tqdm
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.loader import DataLoader
from networkx.algorithms.traversal.breadth_first_search import bfs_edges
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score, cohen_kappa_score
import torch.nn.functional as F
import torch
import neptune
import logging

# Neptune API token
NEPTUNE_KEY = os.environ['NEPTUNE_API_TOKEN']

# Get everything we prepared
def get_everything(data_path):
    # Training Labels
    ## Load entity type labels
    print('Loading entity type labels...')
    ent_type = torch.tensor(np.load(f'{data_path}/ent_type_onehot.npy')) # (num_ent, num_ent_type)

    ## Load center molecule motifs
    print('Loading center molecule motifs...')
    motifs = []
    with open(f'{data_path}/id2motifs.json', 'r') as f:
        id2motifs = json.load(f)
    motif_len = len(id2motifs['0'])
    for i in range(len(ent_type)):
        if str(i) in id2motifs.keys():
            motifs.append(np.array(id2motifs[str(i)]))
        else:
            motifs.append(np.array([0] * motif_len))

    motifs = torch.tensor(np.array(motifs), dtype=torch.long) # (num_ent, motif_len)

    ## Center molecule ids
    center_molecule_ids = torch.tensor([int(key) for key in id2motifs.keys()])

    # Entire Knowledge Graph (MolKG)
    print('Loading entire knowledge graph...')
    G = nx.read_gpickle(f'{data_path}/graph.gpickle')
    G_tg = from_networkx(G)

    # molecule_mask
    print('Loading molecule mask...')
    molecule_mask = torch.tensor(ent_type[:,0][G_tg.edge_index[0]] == 1) # (num_edges,)

    return ent_type, motifs, G_tg, center_molecule_ids, molecule_mask


def load_kge_embeddings(emb_path):
    # Load KGE embeddings
    print('Loading KGE embeddings...')
    with open(f'{emb_path}/entity_embedding_.pkl', 'rb') as f:
        entity_embedding = pickle.load(f)
    with open(f'{emb_path}/relation_embedding_.pkl', 'rb') as f:
        relation_embedding = pickle.load(f)
    
    return entity_embedding.clone().detach(), relation_embedding.clone().detach()


# Get subgraph
def get_subgraph(G_tg, molecule_mask, center_molecule_id, motifs, ent_type):
    nodes, _, _, edge_mask = k_hop_subgraph(int(center_molecule_id), 1, G_tg.edge_index)
    double_mask = molecule_mask * edge_mask
    mask_idx = torch.where(double_mask)[0]
    edge_subgraph = G_tg.edge_subgraph(mask_idx)
    subgraph = edge_subgraph.subgraph(nodes)
    masked_node_ids = edge_subgraph.edge_index[0] # (num_masked_nodes,)
    motif_labels = motifs[center_molecule_id] # (num_masked_nodes, motif_len)
    node_labels = ent_type[masked_node_ids] # (num_masked_nodes, num_ent_type)

    subgraph.masked_node_ids = masked_node_ids
    subgraph.center_molecule_id = torch.where(masked_node_ids == center_molecule_id)[0][0]
    subgraph.motif_labels = motif_labels
    subgraph.node_labels = node_labels

    return subgraph


class Dataset(torch.utils.data.Dataset):
    def __init__(self, G_tg, center_molecule_ids, molecule_mask, motifs, ent_type):
        self.G_tg = G_tg # torch_geometric.data.Data
        self.center_molecule_ids = center_molecule_ids # list of int
        self.molecule_mask = molecule_mask # (num_edges,)
        self.motifs = motifs # (num_ent, motif_len)
        self.ent_type = ent_type # (num_ent, num_ent_type)

    def __len__(self):
        return len(self.center_molecule_ids)
    def __getitem__(self, idx):
        return get_subgraph(
            G_tg=self.G_tg, 
            molecule_mask=self.molecule_mask, 
            center_molecule_id=self.center_molecule_ids[idx], 
            motifs=self.motifs, 
            ent_type=self.ent_type
            )


def get_dataloader(G_tg, center_molecule_ids, molecule_mask, motifs, ent_type, batch_size):
    dataset = Dataset(G_tg, center_molecule_ids, molecule_mask, motifs, ent_type)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size 
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader


def train(model, train_loader, device, optimizer, run=None):
    model.train()
    training_loss = 0
    tot_loss = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, data in pbar:
        pbar.set_description(f'loss: {training_loss}')
        data = data.to(device)
        optimizer.zero_grad()

        # Forward
        edge_class, motif_pred, node_class = model(data.masked_node_ids, data.relation, data.center_molecule_id, data.edge_index)

        motif_labels = data.motif_labels.reshape(int(train_loader.batch_size), int(len(data.motif_labels)/train_loader.batch_size)).float()
        # Loss
        loss, edge_loss, motif_loss, node_class_loss = model.loss(edge_class, motif_pred, node_class, data.rel_label, motif_labels, data.node_labels)

        # Backward
        loss.backward()
        training_loss = loss
        tot_loss += loss
        optimizer.step()
        run["train/step_loss"].append(training_loss)
        run["train/step_edge_loss"].append(edge_loss)
        run["train/step_motif_loss"].append(motif_loss)
        run["train/step_node_class_loss"].append(node_class_loss)

    return tot_loss


def validate(model, val_loader, device, run=None):
    model.eval()
    val_loss = 0
    tot_loss = 0
    y_prob_edge_all, y_prob_motif_all, y_prob_node_all, y_true_edge_all, y_true_motif_all, y_true_node_all = [], [], [], [], [], []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for i, data in pbar:
        pbar.set_description(f'loss: {val_loss}')
        data = data.to(device)

        # Forward
        edge_class, motif_pred, node_class = model(data.masked_node_ids, data.relation, data.center_molecule_id, data.edge_index)

        motif_labels = data.motif_labels.reshape(int(val_loader.batch_size), int(len(data.motif_labels)/val_loader.batch_size)).float()
        # Loss
        loss, edge_loss, motif_loss, node_class_loss = model.loss(edge_class, motif_pred, node_class, data.rel_label, motif_labels, data.node_labels)

        val_loss = loss
        tot_loss += loss
        
        y_prob_edge = F.softmax(edge_class, dim=-1)
        y_prob_motif = F.sigmoid(motif_pred, dim=-1)
        y_prob_node = F.softmax(node_class, dim=-1)

        node_labels = data.node_labels.reshape(int(val_loader.batch_size), int(len(data.node_labels)/val_loader.batch_size)).float()
        rel_label = data.rel_label.reshape(int(val_loader.batch_size), int(len(data.rel_label)/val_loader.batch_size)).float()

        y_true_edge, y_true_motifs, y_true_node = rel_label, motif_labels, node_labels

        y_prob_edge_all.append(y_prob_edge)
        y_prob_motif_all.append(y_prob_motif)
        y_prob_node_all.append(y_prob_node)
        y_true_edge_all.append(y_true_edge)
        y_true_motif_all.append(y_true_motifs)
        y_true_node_all.append(y_true_node)

        run["val/step_loss"].append(val_loss)
        run["val/step_edge_loss"].append(edge_loss)
        run["val/step_motif_loss"].append(motif_loss)
        run["val/step_node_class_loss"].append(node_class_loss)
    
    return tot_loss, y_prob_edge_all, y_prob_motif_all, y_prob_node_all, y_true_edge_all, y_true_motif_all, y_true_node_all


def metric_calculation(y_prob_all, y_true_all, mode='multiclass'):
    if mode == "multilabel":
        y_pred_all = (y_prob_all >= 0.5).astype(int)

        val_pr_auc = average_precision_score(y_true_all, y_prob_all, average="samples")
        val_roc_auc = roc_auc_score(y_true_all, y_prob_all, average="samples")
        val_jaccard = jaccard_score(y_true_all, y_pred_all, average="samples", zero_division=1)
        val_acc = accuracy_score(y_true_all, y_pred_all)
        val_f1 = f1_score(y_true_all, y_pred_all, average="samples", zero_division=1)
        val_precision = precision_score(y_true_all, y_pred_all, average="samples", zero_division=1)
        val_recall = recall_score(y_true_all, y_pred_all, average="samples", zero_division=1)

    elif mode == "multiclass":
        y_pred_all = np.argmax(y_prob_all, axis=-1)
        y_true_all = np.argmax(y_true_all, axis=-1)

        val_pr_auc = 0
        val_roc_auc = roc_auc_score(y_true_all, y_prob_all, multi_class="ovr", average="weighted")
        val_jaccard = cohen_kappa_score(y_true_all, y_pred_all)
        val_acc = accuracy_score(y_true_all, y_pred_all)
        val_f1 = f1_score(y_true_all, y_pred_all, average="weighted")
        val_precision = 0
        val_recall = 0

    return val_pr_auc, val_roc_auc, val_jaccard, val_acc, val_f1, val_precision, val_recall


def detach_numpy(tensor):
    return torch.cat(tensor, dim=0).cpu().detach().numpy()


def train_loop(model, train_loader, val_loader, optimizer, device, epochs, logger=None, run=None, early_stop=5):
    best_roc_auc = 0
    best_f1 = 0
    early_stop_indicator = 0
    for epoch in range(1, epochs+1):
        train_loss = train(model, train_loader, device, optimizer, run=run)
        valid_loss, y_prob_edge_all, y_prob_motif_all, y_prob_node_all, y_true_edge_all, y_true_motif_all, y_true_node_all = validate(model, val_loader, device, run=run)
        
        y_prob_edge_all, y_prob_motif_all, y_prob_node_all, y_true_edge_all, y_true_motif_all, y_true_node_all = detach_numpy(y_prob_edge_all), detach_numpy(y_prob_motif_all), detach_numpy(y_prob_node_all), detach_numpy(y_true_edge_all), detach_numpy(y_true_motif_all), detach_numpy(y_true_node_all)

        edge_val_pr_auc, edge_val_roc_auc, edge_val_jaccard, edge_val_acc, edge_val_f1, edge_val_precision, edge_val_recall = metric_calculation(y_prob_edge_all, y_true_edge_all, mode="multiclass")
        motif_val_pr_auc, motif_val_roc_auc, motif_val_jaccard, motif_val_acc, motif_val_f1, motif_val_precision, motif_val_recall = metric_calculation(y_prob_motif_all, y_true_motif_all, mode="multilabel")
        node_val_pr_auc, node_val_roc_auc, node_val_jaccard, node_val_acc, node_val_f1, node_val_precision, node_val_recall = metric_calculation(y_prob_node_all, y_true_node_all, mode="multiclass")

        if motif_val_roc_auc >= best_roc_auc:
            torch.save(model.state_dict(), f'/data/pj20/molkg/kgnn_epoch_{epoch}.pkl')
            print("best model saved")
            best_roc_auc = motif_val_roc_auc
            early_stop_indicator = 0
            # best_f1 = val_f1
        else:
            early_stop_indicator += 1
            if early_stop_indicator >= early_stop:
                break
        if run is not None:
            run["train/epoch_loss"].append(train_loss)
            run["val/loss"].append(valid_loss)
            run["val/edge_pr_auc"].append(edge_val_pr_auc)
            run["val/edge_roc_auc"].append(edge_val_roc_auc)
            run["val/edge_acc"].append(edge_val_acc)
            run["val/edge/f1"].append(edge_val_f1)
            run["val/edge/precision"].append(edge_val_precision)
            run["val/edge/recall"].append(edge_val_recall)
            run["val/edge/jaccard"].append(edge_val_jaccard)
            run["val/motif_pr_auc"].append(motif_val_pr_auc)
            run["val/motif_roc_auc"].append(motif_val_roc_auc)
            run["val/motif_acc"].append(motif_val_acc)
            run["val/motif/f1"].append(motif_val_f1)
            run["val/motif/precision"].append(motif_val_precision)
            run["val/motif/recall"].append(motif_val_recall)
            run["val/motif/jaccard"].append(motif_val_jaccard)
            run["val/node_pr_auc"].append(node_val_pr_auc)
            run["val/node_roc_auc"].append(node_val_roc_auc)
            run["val/node_acc"].append(node_val_acc)
            run["val/node/f1"].append(node_val_f1)
            run["val/node/precision"].append(node_val_precision)
            run["val/node/recall"].append(node_val_recall)
            run["val/node/jaccard"].append(node_val_jaccard)


        if logger is not None:
            logger.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, Val ROC-AUC: {motif_val_roc_auc:.4f}, Val F1: {motif_val_f1:.4f}, Val Precision: {motif_val_precision:.4f}, Val Recall: {motif_val_recall:.4f}, Val Jaccard: {motif_val_jaccard:.4f}')


def get_logger(lr, hidden_dim, epochs, lambda_):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(f'./training_logs/lr_{lr}_dim_{hidden_dim}_epochs_{epochs}_lambda_{lambda_}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def run():
    # your credentials
    run = neptune.init_run(
        project="patrick.jiang.cs/Gode",
        api_token=NEPTUNE_KEY,
    )  

    params = {
        "lr": 1e-4,
        "hidden_dim": 200,
        "epochs": 100,
        "lambda": "08_15_10"
    }
    logger = get_logger(lr=params['lr'], hidden_dim=params['hidden_dim'], epochs=params['epochs'], lambda_=params['lambda'])
    run["parameters"] = params


    # Data path
    data_path = '../data_process/pretrain_data'
    print('Getting everything prepared...')
    ent_type, motifs, G_tg, center_molecule_ids, molecule_mask = get_everything(data_path)

    # Get dataloader
    train_loader, val_loader = get_dataloader(G_tg, center_molecule_ids, molecule_mask, motifs, ent_type, batch_size=1)

    # Load KGE embeddings
    # return:
    # entity_embedding: (num_ent, emb_dim)
    # relation_embedding: (num_rel, emb_dim)
    emb_path = '/data/pj20/molkg_kge/transe'
    print('Loading KGE embeddings...')
    entity_embedding, relation_embedding = load_kge_embeddings(emb_path)

    # Initialize model
    print('Initializing model...')
    model = KGNN(
        node_emb=entity_embedding,
        rel_emb=relation_embedding,
        num_nodes=ent_type.shape[0],
        num_rels=39,
        embedding_dim=512,
        hidden_dim=200,
        num_motifs=motifs.shape[1],
        lambda_edge=0.8,
        lambda_motif=1.5,
        lambda_mol_class=1
    )

    # Train
    device = torch.device('cuda:4')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 100

    print('Start training !!!')
    train_loop(model, train_loader, val_loader, optimizer, device, epochs, run=run)


if __name__ == '__main__':
    run()
