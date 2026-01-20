import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import nodes_to_graph
from torch.distributions import kl_divergence
import torch.distributions as dist
import pdb
from src.gnn import GNN
from src.modules import *
import math
from src.rvq import ResidualVectorQuantizationVanilla

class IMKGC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_kgs = args.num_kgs
        self.num_entities = args.num_entities
        self.num_relations = args.num_relations
        self.pretrain_dim = args.pretrain_dim

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.device = args.device
        self.criterion = nn.MarginRankingLoss(margin=args.margin, reduction='mean') 
        self.info_nce = ContrastiveLoss(temp=args.temp)
        
        self.entity_embedding_layer = nn.Embedding(self.num_entities, self.entity_dim)
        nn.init.xavier_uniform_(self.entity_embedding_layer.weight)

        self.rel_embedding_layer = nn.Embedding(self.num_relations, self.relation_dim)
        nn.init.xavier_uniform_(self.rel_embedding_layer.weight)
        
        self.relation_prior = nn.Embedding(self.num_relations, 1) 
        nn.init.xavier_uniform_(self.relation_prior.weight)

        self.kg_embedding_layer = nn.Embedding(self.num_kgs, self.entity_dim) 
        nn.init.xavier_uniform_(self.kg_embedding_layer.weight)
            
        self.encoder_KG = GNN(num_kgs=args.num_kgs, in_dim=args.entity_dim, in_edge_dim=args.relation_dim, n_hid=args.encoder_hdim_gnn,
                                out_dim=args.entity_dim, n_heads=args.n_heads, n_layers=args.n_layers_gnn, dropout=args.dropout)
        
        self.layer_norm_gnn = nn.LayerNorm(self.entity_dim)
        self.msib = MuSigmaEncoder(args.entity_dim, args.entity_dim, args)

        self.decoder = TransE(device=self.device)

        self.rvq = ResidualVectorQuantizationVanilla(num_codebooks=self.args.reason_step, codebook_size=int(self.num_relations*self.args.codebook_ratio), codebook_dim=self.relation_dim, commitment_cost=self.args.commit_loss)


    def forward_GNN_embedding(self, graph_input_list, kg_index):
        '''
        graph_input_list: [5, ]
        '''
        x_gnn_output_all = []
        z_all = []
        x_gnn_dist_all = []

        for i, graph_input in enumerate(graph_input_list):
            x_features = self.entity_embedding_layer(graph_input.x) # [num_nodes, entity_dim]
            edge_index = graph_input.edge_index # [2, num_edges]
            edge_kg_index = graph_input.edge_kg_index # [2, num_edges]
            edge_beta_r = self.relation_prior(graph_input.edge_attr) # [num_edges, 1]
            edge_relation_embedding = self.rel_embedding_layer(graph_input.edge_attr) # [num_edges, relation_dim]
            x_gnn_output = self.encoder_KG(x_features, edge_index, edge_kg_index, edge_beta_r, edge_relation_embedding, graph_input.y, graph_input.num_size)
            x_gnn_output = self.layer_norm_gnn(x_gnn_output)
            x_gnn_output_all.append(x_gnn_output)

            x_gnn_dist = self.msib(x_gnn_output) # variational
            x_gnn_dist_all.append(x_gnn_dist)
            
            if self.training:
                z = x_gnn_dist.rsample()
                z_all.append(z)
            else:
                z = x_gnn_dist.sample()
                z_all.append(z)

        x_gnn_output_all = torch.stack(x_gnn_output_all) # [num_kgs, batch_size, out_dim]
        x_gnn_output_all = x_gnn_output_all.permute(1, 0, 2) # [batch_size, num_kgs, out_dim]
        
        x_out = torch.mean(x_gnn_output_all, dim=1) # [b, e]
        kld = 0.
        return  x_out, {'z':z_all, 'x_gnn_dist':x_gnn_dist_all, 'kld':kld}


    def mutual_info(self, experts):
        num = len(experts)
        idx1, idx2 = random.sample(range(num), k=2)
        expert1, expert2 = experts[idx1], experts[idx2]
        return self.info_nce(expert1, expert2)


    def forward_kg(self, h_graph, sample, t_graph, t_neg_graph, kg_index):
        # h_graph, sample：[200, 3], t_graph, t_neg_graph: subgraph, kg_index: id of KG
        h, h_experts = self.forward_GNN_embedding(h_graph, kg_index) # [batch_size, d]
        r = self.rel_embedding_layer(sample[:, 1]) # [batch_size, d]
        t, t_experts = self.forward_GNN_embedding(t_graph, kg_index) # [batch_size, d]
        t_neg, t_neg_experts = self.forward_GNN_embedding(t_neg_graph, kg_index) # [batch_size, d]

        if self.args.v_rel == 'RVQ':
            quantized_out, all_indices, vq_loss = self.rvq(r)
            r = quantized_out
        elif self.args.v_rel == 'RVQN':
            quantized_out, all_indices, vq_loss = self.rvq(r)
            r = quantized_out
        else:
            vq_loss = torch.tensor([0.]).cuda()

        r = r.unsqueeze(1)
        h = h.unsqueeze(1)
        t = t.unsqueeze(1)
        t_neg = t_neg.unsqueeze(1)

        if self.num_kgs > 1:
            info_contrastive = self.mutual_info(h_experts['z']) + self.mutual_info(t_experts['z']) + self.mutual_info(t_neg_experts['z'])
        else:
            info_contrastive = torch.tensor([0.]).cuda()

        bs = h.size(0)
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        loss_assist_kg = torch.tensor([0.]).cuda()
        loss_assist_kld = torch.tensor([0.]).cuda()

        loss_cur_kg = torch.tensor([0.]).cuda()
        normal_dist = torch.distributions.Normal(torch.zeros_like(h_experts['z'][0]), torch.ones_like(h_experts['z'][0]))
        for i in range(self.num_kgs):
            h_z = h_experts['z'][i].unsqueeze(1)
            t_z = t_experts['z'][i].unsqueeze(1)
            t_neg_z = t_neg_experts['z'][i].unsqueeze(1)
            pos_loss, neg_loss = self.decoder(h_z, r, t_z, t_neg_z)
            kg_loss = self.criterion(pos_loss.expand(bs, bs).reshape(-1), neg_loss.transpose(-1,-2).expand(bs, bs).reshape(-1), target) # margin loss

            kld_loss = (kl_divergence(h_experts['x_gnn_dist'][i], normal_dist).sum(-1).mean(dim=0) + kl_divergence(t_experts['x_gnn_dist'][i], normal_dist).sum(-1).mean(dim=0) + kl_divergence(t_neg_experts['x_gnn_dist'][i], normal_dist).sum(-1).mean(dim=0))/3

            if i == kg_index:
                loss_cur_kg = kg_loss
            else:
                loss_assist_kg += kg_loss
            
            loss_assist_kld += kld_loss

        pos_loss, neg_loss = self.decoder(h, r, t, t_neg)
        kg_loss = self.criterion(pos_loss.expand(bs, bs).reshape(-1), neg_loss.transpose(-1,-2).expand(bs, bs).reshape(-1), target) # margin loss
        total_loss = {'kg_loss': kg_loss, 'kld_loss':loss_assist_kld, 'kg_cur_loss':loss_cur_kg, 'kg_assist_loss':loss_assist_kg, 'info_contrastive':info_contrastive,  'vq_loss':vq_loss}
        return total_loss
    
    def predict_r_embedding(self, r):
        r = self.rel_embedding_layer(r)
        quantized_out, all_indices, vq_loss = self.rvq(r)
        r = quantized_out
        return r
        
    
    def predict(self, h_emb, r, z=None):
        return self.decoder.predict(h_emb, r, z)

    def predict_candidate(self, c, z=None):
        return self.decoder.predict_candidate(c, z)

    def predict_score_fuc(self):
        return self.decoder.define_score

    
class TransE(nn.Module):
    def __init__(self, device):
        super(TransE, self).__init__()
        self.device = device

    def project_t(self, hr):
        return hr[0] + hr[1]

    def define_score(self, t_true_pred, d_r=None):
        t_true = t_true_pred[0]
        t_pred = t_true_pred[1]
        # t_true: [b, 1, e]
        # t_pred: [b, 1, e] or [b, a, e]
        return torch.norm(t_true - t_pred + 1e-8, dim=2) # [b,1] or [b,a]

    def forward(self, h, r, t, t_neg): 
        '''
        h,r,t,t_neg: [b,1,e]
        t_neg: []
        '''
        projected_t = self.project_t([h, r]) # h + r
        pos_loss = self.define_score([t, projected_t]) # norm(t - (h+r)) [batch_size, 1]
        neg_loss = self.define_score([t_neg, projected_t]) # norm(t - t_neg) [batch_size, num_neg] or [batch_size, 1]
        return pos_loss, neg_loss

    def predict(self, h_emb, r, z=None):
        # h_emb: [batch_size, d]
        # r: [batch_size, d]
        h = h_emb.unsqueeze(1) # [batch_size, 1, d]
        r = r.unsqueeze(1) # [batch_size, 1, d]
        projected_t = self.project_t([h, r]) # [b, 1, e]
        return projected_t
    
    def predict_candidate(self, c, z=None):
        return c.unsqueeze(0) # [1,a,e]
    