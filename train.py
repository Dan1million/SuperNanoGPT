import torch;
from languageModel.bigramLanguageModel import BigramLanguageModel

# Parameters
block_size = 64 # Maximum context length
batch_size = 256 # Maximum number of parallel executions
max_iters = 5000 # Number of learning iterations
eval_interval = 300 # Number of iterations break before outputing evaluation
learning_rate = 3e-4 # Learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Device to run the language model on
eval_iters = 200 # Number of batches to evaluate in the evaluation step
n_embd = 384 # Number of embedding dimensions to use for embeddings
n_layer = 6 # Number of transformers used in the language model
n_heads = 6 # Number of heads in each multi-headed attention block
dropout = 0.2 # Dropout percentage to maintain evolution


# Read in the data set
with open('input.txt', 'r', encoding='utf-8') as f :
    text = f.read()


# Tokenizaiton --> Each unique character is a token
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create mapping to convert tokens to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda string: [stoi[c] for c in string] # Encoder to convert string to a list of integers representing the characters
decode = lambda list: ''.join([itos[i] for i in list]) # Decoder to convert a list of integers to it's string equivalent


# Split the dataset into a training dataset and validation dataset
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n] # 90% to train
val_data = data[n:] # 10% to test


def get_batch(dataset):
    """
        Randomly samples the training or validation data set

        Args:
            dataset string: 'train' if usign the training data set, 'val' if using the validation training set
    """
    data = train_data if dataset == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Create a random offset for a batch
    x = torch.stack([data[i:i+block_size] for i in ix]) # Create block of characters chunk 'training data for tensor'
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # Create off by one block of characters chunk 'correct predictions'
    x, y = x.to(device), y.to(device) # Move the data set to the device
    return x, y

@torch.no_grad() # we will not call backward for back propogation in this function --> reduces memory footprint
def estimate_loss():
    """
        Esitmates the current training data loss and validation data loss
    """
    out = {}
    model.eval()
    for dataset in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(dataset)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[dataset] = losses.mean()
    model.train()
    return out

# Create the bigram language model and offload to GPU if possible
model = BigramLanguageModel(block_size, device, dropout, n_embd, n_heads, n_layer, vocab_size)
m = model.to(device)

# Output number of parameters to command line
print(sum(p.numel() for p in m.parameters()), 'Parameters')

# Using the AdamW pytorch optimizer --> performs the gradient descent calculation gradient descent
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0: # Occasionaly output current state of training
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample the training data
    xb, yb = get_batch('train')

    # Evaluate the loss and perform gradient descent
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Create input vector representing input token index
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Generate new tokens based on our trained ML model!
print(decode(m.generate(idx = context, max_new_tokens=500)[0].tolist()))
