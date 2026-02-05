import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    """
        Multi-headed self attention block that supports configuring the number of heads,
        size of the heads, number of embeddings (dimensions in the embedding), and a 
        dropout value (helps negate overfitting).

        Implementation of "Attention Is All You Need" pg. 3 with only the Decoder block

        Note: This is a decoder only implmeentation since we are not working with a data
        set that requires any kind of additional encoding.
    """

    def __init__(self, block_size, dropout, n_embd, n_heads):
        """
            Initializes a multi-headed attention block
        
            Args:
                block_size int: Maximum Number of tokens processed at once
                dropout float32: percentage of results to "dropout" to maintain evolution --> See: https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b4
                n_embd int: number of dimensions in the embedding
                n_heads int: Number of heads in the attention block
        """
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([Head(block_size, dropout, head_size, n_embd) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        """
            Concatenates the multi headed outputs into a single tensor. Projects the output
            To allow the heads to forward communicate with eachother.

            Args:
                x tensor: the input tensor for the multi-headed self-attention block
        """
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        return out

class Head(nn.Module):
    """
        Single head of self-attention as outlined in "Attention Is All You Need" pg. 4
        Performs the scaled dot-product attention operation as outlined in the 
        Attention(Q, K, V) equation
    """

    def __init__(self, block_size, dropout, head_size, n_embd):
        """
            Initializes a single head of a self attention

            Args:
                block_size int: Maximum Number of tokens processed at once
                dropout float32: percentage of results to "dropout" to maintain evolution
                head_size int: the size of this individual attention head
                n_embd int: number of dimensions in the embedding
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # initializes the 1's triangle that enforces only using context previous to the current token
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
            The real bread and butter of the self attention block. Performs the now
            famous Attention(Q, K, V) calculation

            Args:
                x tensor: the input tensor for the self-attention block
        """
        B,T,C = x.shape # (B,T,C) --> (Batch Size, Sequence Length, Embedding Size)
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # Add head dimension for scaled_dot_product_attention: (B, 1, T, head_size)
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        # Compute attention using PyTorch's optimized scaled dot-product attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )
        
        # Remove head dimension: (B, 1, T, head_size) -> (B, T, head_size)
        out = out.squeeze(1)
        return out
