import torch
import torch.nn as nn
from torch.nn import functional as F
from languageModel.transformerBlock import TransformerBlock

class BigramLanguageModel(nn.Module):
    """
        A simple bigram language model that predicts the next token in the sequence
        based on the immediately preceding token.
    """

    def __init__(self, block_size, device, dropout, n_embd, n_heads, n_layer, vocab_size):
        """
            Initializer for the bigram language model implementation. Initializes the
            embedding table, positional embedding table, transformer blocks, layer
            normalization, andembedding score calculator (outputs logits)

            Args:
                block_size int: Maximum Number of tokens processed at once
                device string: cuda if Cuda is supported, CPU otherwise
                dropout float32: percentage of results to "dropout" to maintain evolution --> See: https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b4
                n_embd int: number of dimensions in the embedding
                n_heads int: Number of heads in each multi-headed attention block
                n_layer: Number of transformers in the language model
                vocab_size: the size of the token vocabulary 
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(block_size, dropout, n_embd, n_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        self.device = device
    
    def forward(self, idx, targets=None):
        """
            Calcualtes the logits value and loss value based on the actual calculated tensor
            and the expected result tensor

            Args:
                idx tensor: The actual tensor result from the current training iteration
                targets tensor: The expected tesnor result based on the trained data
        """
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx) # Retrieve the token embedding for idx from the embedding table
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # # Retrieve the position embedding value from the position embedding table
        x = token_emb + pos_emb # Combines the semantic (token embedding) and positional (position embedding) to form a tensor that encodes both the positional and semantic information
        x = self.blocks(x) # Provide the embedding to the transformer blocks
        x = self.ln_f(x) # Normalize the layers
        logits = self.lm_head(x) # Calculate the logits

        if targets is None: # Generating new tokens
            loss = None
        else: # Training
            # PyTorch wants channels to be the second dimension for this operation --> (B, C, T)
            B, T, C = logits.shape # Use dot products to prepare the outputs for the loss calculation
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # Calculate the loss using the cross entropy process --> Combination of softmax and negative log probability
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
            Generates new tokens based on the trained bigram language model

            Args:
                idx tensor: The tensor that "kicks off" generation of new tokens
                max_new_tokens int: The number of new tokens to generate
        """

        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # Get logits predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]
            # Apply softmax --> get probabilities for each token
            probs = F.softmax(logits, dim=1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the return sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
