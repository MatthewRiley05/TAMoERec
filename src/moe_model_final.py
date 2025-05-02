#moe model final

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv
from trainers import CoSeRecTrainer
from datasets import RecWithContrastiveLearningDataset
from utils import EarlyStopping, get_user_seqs, check_path, set_seed
from models import OfflineItemSimilarity, OnlineItemSimilarity, SASRec
from GAT import GATRec
from GRU4recs import GRU4Recs



class MoERecommender(nn.Module):
    def __init__(self, args):
        super(MoERecommender, self).__init__()
        
        self.args = args
        self.n_experts = 2
        self.hidden_size = args.hidden_size
        self.item_size = args.item_size
        self.max_seq_length = args.max_seq_length
        self.dropout_prob = args.hidden_dropout_prob
        
        self.item_embedding = nn.Embedding(self.item_size, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        
        self.experts = nn.ModuleList([
            GATRec(args),
            GRU4Recs(args),  
            SASRec(args)          
        ])
        

        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.experts)),
            nn.Softmax(dim=-1)
        )
        
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.current_gate_weights = None
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, item_seq):

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)
        
        item_emb = self.item_embedding(item_seq)
        position_emb = self.position_embedding(position_ids)
        
        input_emb = item_emb + position_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        
        gate_input = input_emb.mean(dim=1)  
        gate_weights = self.gate(gate_input)
        self.current_gate_weights = gate_weights   

        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(item_seq)  
            expert_outputs.append(expert_output)
        
        # Combine expert outputs using gate 
        combined_output = torch.zeros_like(expert_outputs[0])
        for i, expert_output in enumerate(expert_outputs):

            weight = gate_weights[:, i].view(-1, 1, 1)
            combined_output += weight * expert_output
            
        aux_loss = self.compute_load_balancing_loss(gate_weights) if self.training else 0
        
        return combined_output, aux_loss
    
    def compute_load_balancing_loss(self, gate_weights):

        expert_usage = gate_weights.mean(0)
        target_usage = torch.ones_like(expert_usage) / len(self.experts)

        aux_loss = F.kl_div(
            expert_usage.log(), 
            target_usage, 
            reduction='batchmean'
        )
        return aux_loss
    
    def get_moe_weights(self):

        if self.current_gate_weights is not None:
            # Return avg weights per batch
            return self.current_gate_weights.mean(dim=0).detach().cpu().numpy() 
        return None