import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU4Recs(nn.Module):

    def __init__(self, args):
        super(GRU4Recs, self).__init__()

        self.hidden_size = args.hidden_size  
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.item_size = args.item_size
        self.max_seq_length = args.max_seq_length
        self.args = args
        
        self.num_gru_layers = getattr(args, 'num_gru_layers', 2)
        
        self.item_embedding = nn.Embedding(self.item_size, self.hidden_size, padding_idx=0)
        
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_gru_layers,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.zero_()

    def forward(self, item_seq):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        
        output, _ = self.gru(item_emb)
        
        return output