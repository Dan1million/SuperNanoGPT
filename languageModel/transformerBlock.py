import torch.nn as nn
from languageModel.head import MultiHeadAttention

class TransformerBlock(nn.Module):
    """
        Implementation of a decode only transformer block as outlined in "Attention Is All You Need"
    """

    def __init__(self, block_size, dropout, n_embd, n_heads):
        """
            Initializes a decode only transformer block with feed-forward and layer-normalization
            capabilities.

            Args:
                block_size int: Maximum Number of tokens processed at once
                dropout float32: percentage of results to "dropout" to maintain evolution --> See: https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b4
                n_embd int: number of dimensions in the embedding
                n_heads int: Number of heads in the multi-headed attention block
        """
        super().__init__()
        self.sa = MultiHeadAttention(block_size, dropout, n_embd, n_heads)
        self.ffwd = FeedForward(dropout, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        """
            Performs the Multi-Head Attantion step and the preceding and proceeding layer normalization

            Args:
                x tensor: the input tensor for the transformer
        """
        x = x + self.sa(self.ln1(x)) # Performs the layer normalization step before the Multi-Head Attention step
        x = x + self.ffwd(self.ln2(x)) # Performs the layer normalization step after the Multi-Head Attention step
        return x

class FeedForward(nn.Module):
    """
        Simple feed forward implementation. Ensures that the context from previous tokens in
        the batch are forwarded to the tokens ahead.
    """

    def __init__(self, dropout, n_embd):
        """
            Initializes a simple feed forward with ReLU and dropout. Explained on pg 5. section 3.3
            in "Attention Is All You Need"

            Args:
                dropout float32: percentage of results to "dropout" to maintain evolution
                n_embd int: number of dimensions in the embedding
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
            Performs the feed forward operation

            Args:
                x tensor: the input tensor for the feed forward operation
        """
        return self.net(x)
