import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # Linear transformation
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention mechanism
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        # Initialize using normal distribution
        nn.init.normal_(self.W.weight, std=0.02)
        nn.init.normal_(self.a.weight, std=0.02)
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # Apply linear transformation
        h = self.W(input)  # [batch_size, seq_len, out_features]
        
        # Create attention inputs
        batch_size, N, _ = h.size()
        
        # Repeat tensors for attention computation
        a_input = torch.cat([h.repeat_interleave(N, dim=1), 
                           h.repeat(1, N, 1)], dim=2)
        a_input = a_input.view(batch_size, N, N, 2 * self.out_features)
        
        # Compute attention coefficients
        e = self.leakyrelu(self.a(a_input).squeeze(-1))
        
        # Mask attention coefficients using adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, h)
        
        return h_prime

class GATRec(nn.Module):

    def __init__(self, args):
        super(GATRec, self).__init__()
        
        # load parameters info
        self.hidden_size = args.hidden_size
        self.item_size = args.item_size
        self.max_seq_length = args.max_seq_length
        self.dropout_prob = getattr(args, 'hidden_dropout_prob', 0.5)
        
        # define layers
        self.item_embedding = nn.Embedding(self.item_size, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        # GAT layer
        self.gat = GATLayer(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            dropout=self.dropout_prob
        )
        
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    #this is supposed to be the forward function of the GATRec model but named like this for consistency with the trainer
    def forward(self, item_seq):

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)

        item_emb = self.item_embedding(item_seq)
        position_emb = self.position_embedding(position_ids)
        
        # merge embeddings
        input_emb = item_emb + position_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        
        # adjacency matrix 
        batch_size, seq_len = item_seq.size()
        adj = torch.ones(batch_size, seq_len, seq_len, device=item_seq.device)
        
        output = self.gat(input_emb, adj)
        
        return output
    

    #testing if saved