import os
import math
import argparse
import copy
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import dgl
from dgl.data.utils import save_graphs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from .model import *
from .utils import *

from .graphmask.moving_average import MovingAverage
from .graphmask.lagrangian_optimization import LagrangianOptimization

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)


class GodeKGNN:

    def __init__(self, data, exp_name='Gode', device='cuda:0'):
        self.device = torch.device(device=device)
        self.data = data
        self.G = data.G
        self.df, self.df_train, self.df_valid, self.df_test = data.df, data.df_train, data.df_valid, data.df_test
        self.data_folder = data.data_folder


    def model_initialize(
        self, 
        input_dim=128, 
        hidden_dim=128, 
        output_dim=128,
        exp_lambda=0.7,
        num_walks=200,
        walk_mode='bit',
        path_length=2,
        ):
        self.G = self.G.to('cpu')
        self.G = initialize_node_embedding(self.G, input_dim)
        self.g_valid_pos, self.g_valid_neg = evaluate_graph_construct(self.df_valid, self.G, "fix_dst", 1, self.device)
        self.g_test_pos, self.g_test_neg = evaluate_graph_construct(self.df_valid, self.G, "fix_dst", 1, self.device)

        self.model = HeteroRGCN(
            self.G,
            in_size=input_dim,
            hidden_size=hidden_dim,
            out_size=output_dim,
            exp_lambda=exp_lambda,
            proto=True,
            proto_num=5,
            attention=False,
            sim_measure='all_nodes_profile',
            agg_measure='rarity',
            num_walks=num_walks,
            walk_mode=walk_mode,
            path_length=path_length,
            data_folder=self.data_folder,
            device=self.device,
        ).to(self.device)

        self.best_model = self.model

    def pretrain(
        self, 
        n_epoch=1, 
        learning_rate=1e-3,
        batch_size=1024,
        ):

        self.G = self.G.to('cpu')
        train_eid_dict = {etype: self.G.edges(form = 'eid', etype =  etype) for etype in self.G.canonical_etypes}
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

        print('edge dataloader')
        dataloader = dgl.dataloading.EdgeDataLoader(
            self.G, train_eid_dict, sampler,
            negative_sampler=Minibatch_NegSampler(self.G, 1, 'fix_dst'),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0)

        print('optimizer')
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        print('Start pre-training with #param: %d' % (get_n_params(self.model)))

        for epoch in range(n_epoch):

            for step, (nodes, pos_g, neg_g, blocks) in enumerate(dataloader):

                blocks = [i.to(self.device) for i in blocks]
                pos_g = pos_g.to(self.device)
                neg_g = neg_g.to(self.device)
                pred_score_pos, pred_score_neg, pos_score, neg_score = self.model.forward_minibatch(pos_g, neg_g, blocks, self.G, mode = 'train', pretrain_mode = True)

                scores = torch.cat((pos_score, neg_score)).reshape(-1,)
                labels = [1] * len(pos_score) + [0] * len(neg_score)

                loss = F.binary_cross_entropy(scores, torch.Tensor(labels).float().to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.weight_bias_track:
                    self.wandb.log({"Pretraining Loss": loss})

                if step % 20 == 0:
                    # pretraining tracking...
                    auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc = get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, self.G, True)
                    
                    if self.weight_bias_track:
                        temp_d = get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, "Pretraining")
                        temp_d.update({"Pretraining LR": optimizer.param_groups[0]['lr']})
                        self.wandb.log(temp_d)
                    
                    print('Epoch: %d Step: %d LR: %.5f Loss %.4f, Pretrain Micro AUROC %.4f Pretrain Micro AUPRC %.4f Pretrain Macro AUROC %.4f Pretrain Macro AUPRC %.4f' % (
                        epoch,
                        step,
                        optimizer.param_groups[0]['lr'], 
                        loss.item(),
                        micro_auroc,
                        micro_auprc,
                        macro_auroc,
                        macro_auprc
                    ))
        self.best_model = copy.deepcopy(self.model)







