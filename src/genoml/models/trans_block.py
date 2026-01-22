import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

class SelfAttention(nn.Module):
    def __init__(self, d_embed, n_heads, dropout_rate=0.1, use_position_embedding=True):
        super().__init__()
        assert d_embed % n_heads == 0, "d_embed must be divisible by n_heads"
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.dropout_rate = dropout_rate
        self.use_position_embedding = use_position_embedding

        if self.use_position_embedding:
            self.rotary_emb = RotaryEmbedding(dim=self.d_head)

        self.q_linear = nn.Linear(d_embed, d_embed)
        self.k_linear = nn.Linear(d_embed, d_embed)
        self.v_linear = nn.Linear(d_embed, d_embed)
        self.out_linear = nn.Linear(d_embed, d_embed)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, d_embed = x.shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)
        q = self.q_linear(x).view(interim_shape).transpose(1, 2)
        k = self.k_linear(x).view(interim_shape).transpose(1, 2)
        v = self.v_linear(x).view(interim_shape).transpose(1, 2)

        if self.use_position_embedding:
            # q.shape = k.shape = (batch_size, n_heads, seq_len, d_head)
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=2)

        attn_scores = torch.einsum('b h q d, b h k d -> b h q k', q, k) / math.sqrt(self.d_head)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum('b h q k, b h k d -> b h q d', attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_embed)
        output = self.out_linear(attn_output)
        return output


class TransBlock(nn.Module):
    def __init__(self, d_embed, n_heads, d_mlp, dropout_rate=0.1, bias=False, use_position_embedding=True):
        super().__init__()

        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.dropout_rate = dropout_rate
        self.use_position_embedding = use_position_embedding

        self.attn = SelfAttention(
            d_embed, 
            n_heads, 
            dropout_rate, 
            use_position_embedding
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(d_embed, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_embed),
        )

        self.layer_norm1 = nn.LayerNorm(d_embed)
        self.layer_norm2 = nn.LayerNorm(d_embed)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, x, mask=None):
        # pre LN
        x_norm = self.layer_norm1(x)
        attn = self.attn(x_norm, mask)
        x = x + self.dropout1(attn)

        x_norm = self.layer_norm2(x)
        mlp = self.mlp(x_norm)
        x = x + self.dropout2(mlp)
        return x
