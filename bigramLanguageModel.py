import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        # Create Batch * time * channels tensor of logits
        logits = self.token_embedding_table(idx) # (B, T, C) (4, 8, 65)

        if targets is None:
            loss = None
        else:
            # pytorch wants channels to be the second dimension for this operation --> (B, C, T)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # Calculate how well we are calculating targets based on the logits 
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # Take (B, T) and generate new tokens up to max_new_tokens length
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            # apply softmax --> get probabilities for each token
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
