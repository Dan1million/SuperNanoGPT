import argparse
import json
import os
import torch
from datetime import datetime
from torch.amp import autocast
from torch.amp import GradScaler
from datasets import load_dataset
from languageModel.bigramLanguageModel import BigramLanguageModel
from tokenizer.tokenizer import Tokenizer

# /////////////////////////////////////////////////////
# Parse command line arguments
# /////////////////////////////////////////////////////
parser = argparse.ArgumentParser(description="Train a Bigram Language Model")
parser.add_argument('--config', type=str, default='config/config.json', help='Path to the configuration file')
parser.add_argument('--vocab', type=str, default=None, help='Path to a pre-built vocabulary file')
parser.add_argument('--output', type=str, default=f'savedResults/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}', help='Directory to save the trained model, cofiguration, and checkpoints')
parser.add_argument('--resume', type=str, default=None, help='Resume training from the latest checkpoint. Argument is the path to the checkpoint directory.')
args = parser.parse_args()

print("Welcome to Super GPT Nano Training!")
print("Training Configuration:", json.dumps(vars(args), indent=4))


# /////////////////////////////////////////////////////
# Verify that checkpoint exists if resuming training
# /////////////////////////////////////////////////////
from_scratch = True
checkpoint_path = None
if args.resume is not None:
    if os.path.exists(args.resume):
        checkpoint_path = args.resume
        from_scratch = False
    else:
        raise ValueError(f"Checkpoint path {args.resume} does not exist.")


# /////////////////////////////////////////////////////
# Load parameters from configuration file
# /////////////////////////////////////////////////////
with open(args.config, 'r') as configuration:
    config_data = json.load(configuration)

batch_size = config_data['batch_size'] # Maximum number of parallel executions
block_size = config_data['block_size'] # Maximum context length
checkpoint_interval = config_data['checkpoint_interval'] # Number of iterations between saving model checkpoints
eval_interval = config_data['eval_interval'] # Number of iterations break before outputing evaluation
eval_iters = config_data['eval_iters'] # Number of batches to evaluate in the evaluation step
learning_rate = config_data['learning_rate'] # Learning rate
max_iters = config_data['max_iters'] # Number of learning iterations
n_embd = config_data['n_embd'] # Number of embedding dimensions to use for embeddings
n_layer = config_data['n_layer'] # Number of transformers used in the language model
n_heads = config_data['n_heads'] # Number of heads in each multi-headed attention block
dropout = config_data['dropout'] # Dropout percentage to maintain evolution
use_openwebtext = config_data.get('use_openwebtext', False) # Whether to use OpenWebText dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Device to run the language model on


# /////////////////////////////////////////////////////
# Load existing tokenizer vocabulary or build a new one
# /////////////////////////////////////////////////////
if args.vocab and os.path.exists(args.vocab):
    print(f'Loading pre-built vocabulary from {args.vocab}')
    tokenizer = Tokenizer.load_vocab(args.vocab)
else:
    if use_openwebtext:
        print('Building tokenizer vocabulary from OpenWebText sample')
        ds = load_dataset("Skylion007/openwebtext", split='train')
        sample_size = min(100000, len(ds))
        sample_text = ' '.join([ds[i]['text'] for i in range(sample_size)])
        tokenizer = Tokenizer(sample_text)
        # Save vocab immediately after building it
        vocab_cache_path = 'vocab_openwebtext.json'
        tokenizer.save_vocab(vocab_cache_path)
        print(f'Vocabulary saved to {vocab_cache_path} for future use (use --vocab {vocab_cache_path} to reuse)')
    else:
        dataset_path = f'datasets/{config_data["dataset"]}'
        print(f'Building tokenizer vocabulary from {dataset_path}')
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokenizer = Tokenizer(text)


# /////////////////////////////////////////////////////
# Load OpenWebText dataset or custom dataset
# /////////////////////////////////////////////////////
if use_openwebtext:
    print('Loading OpenWebText dataset from Hugging Face')
    ds = load_dataset("Skylion007/openwebtext", split='train', streaming=True)
    
    # Process in chunks to avoid memory issues
    print('Tokenizing OpenWebText dataset in chunks')
    chunk_size = 100000  # Process 100k examples at a time
    max_tokens = 100_000_000  # Limit to 100M tokens (~200MB) to avoid memory issues
    
    all_tokens = []
    for i, example in enumerate(ds):
        if i % 10000 == 0:
            print(f'Tokenized {i} examples, {len(all_tokens):,} tokens so far')
        
        tokens = tokenizer.encode(example['text'])
        all_tokens.extend(tokens)
        
        # Stop if we've collected enough tokens
        if len(all_tokens) >= max_tokens:
            print(f'Reached target of {max_tokens:,} tokens')
            break
    
    data = torch.tensor(all_tokens, dtype=torch.long)
    del all_tokens
else:
    dataset_path = f'datasets/{config_data["dataset"]}'
    print(f'Loading dataset from {dataset_path}')
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
print(f'Dataset loaded: {len(data):,} total tokens')

# Save the tokenizer vocabulary to the output directory
tokenizer.save_vocab(f'{args.output}/vocab.json')
print(f'Tokenizer vocabulary saved to {args.output}/vocab.json')


# /////////////////////////////////////////////////////
# Split dataset into training and validation sets
# /////////////////////////////////////////////////////
n = int(0.9*len(data))  # 90% train, 10% val for larger datasets
train_data = data[:n]
val_data = data[n:]
print(f'Dataset split: {tokenizer.vocab_size()} unique characters, {len(train_data):,} training tokens, {len(val_data):,} validation tokens.')


# /////////////////////////////////////////////////////
# Helper functions
# /////////////////////////////////////////////////////
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
        Estimates the current training data loss and validation data loss
    """
    out = {}
    model.eval()
    for dataset in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(dataset)
            with autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[dataset] = losses.mean()
    model.train()
    return out


# //////////////////////////////////////////////////////////////////////
# Initialize the Bigram Language Model from a checkpoint or from scratch
# //////////////////////////////////////////////////////////////////////
if not from_scratch:
    # Load model checkpoint
    print(f"Resuming training from checkpoint at {checkpoint_path}")
    checkpoint = torch.load(os.path.join(checkpoint_path, 'result.pt'), weights_only=True)
    model = BigramLanguageModel(block_size, device, dropout, n_embd, n_heads, n_layer, tokenizer.vocab_size())
    model.load_state_dict(checkpoint)
    model = model.to(device)
else:
    # Create the bigram language model from scratch
    print("Creating model from scratch")
    model = BigramLanguageModel(block_size, device, dropout, n_embd, n_heads, n_layer, tokenizer.vocab_size())

# Move to GPU if available
m = model.to(device)


# /////////////////////////////////////////////////////
# Initialize the Optimizer and Learning Rate Scheduler
# /////////////////////////////////////////////////////
warmup_iters = int(0.1*max_iters)
cosine_iters = max_iters - warmup_iters

# Load the Optimizer and Scheduler from checkpoint if resuming
if not from_scratch:
    print("Loading Optimizer and Scheduler state from checkpoint")
    # Load Optimizer state from checkpoint
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    optimizer_state = torch.load(os.path.join(checkpoint_path, 'optimizer.pt'))
    optimizer.load_state_dict(optimizer_state)
    # Load Scheduler state from checkpoint
    scheduler_state = torch.load(os.path.join(checkpoint_path, 'scheduler.pt'))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_iters)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iters])
    scheduler.load_state_dict(scheduler_state)
else:
    print("Creating Optimizer and Scheduler from scratch")
    # Create Optimizer and Scheduler from scratch
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_iters)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iters])


# ///////////////////////////////////////////////////////
# Initialize the Grad Scaler for mixed precision training
# ///////////////////////////////////////////////////////
scaler = GradScaler(device=device, enabled=(device == 'cuda'))


# ///////////////////////////////////////////////////////
# Final logging and output before starting training loop
# ///////////////////////////////////////////////////////
current_iter = 0
if not from_scratch:
    current_iter = int(checkpoint_path.split('_')[-1]) # Extract current iteration from checkpoint path
    print(f'Resuming from iteration {current_iter}')

# Output number of parameters to command line
print("Number of parameters:", sum(p.numel() for p in m.parameters()), 'Parameters')
print(f"Mixed precision training: {'Enabled' if device == 'cuda' else 'Disabled (CPU mode)'}")
os.makedirs(args.output, exist_ok=True) # Setup output directory
os.makedirs(f'{args.output}/checkpoints', exist_ok=True) # Setup checkpoints directory


# ///////////////////////////////////////////////////////
# Training Loop
# ///////////////////////////////////////////////////////
for iter in range(current_iter, max_iters):
    if iter % eval_interval == 0: # Occasionaly output current state of training
        losses = estimate_loss()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {scheduler.get_last_lr()[0]:.6f}")

    if iter % checkpoint_interval == 0: # Save model checkpoints
        os.makedirs(f'{args.output}/checkpoints/iteration_{iter}', exist_ok=True)
        torch.save(m.state_dict(), f'{args.output}/checkpoints/iteration_{iter}/result.pt')
        torch.save(optimizer.state_dict(), f'{args.output}/checkpoints/iteration_{iter}/optimizer.pt')
        torch.save(scheduler.state_dict(), f'{args.output}/checkpoints/iteration_{iter}/scheduler.pt')
        with open(f'{args.output}/checkpoints/iteration_{iter}/config.json', 'w') as json_file:
            json.dump(config_data, json_file, indent=4)

    # Sample the training data
    xb, yb = get_batch('train')

    # Evaluate the loss and perform gradient descent with mixed precision
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
        logits, loss = m(xb, yb)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer) # Unscale before gradient clipping
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0) # Gradient clipping to avoid exploding gradients
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()


# ///////////////////////////////////////////////////////
# Save Final Result Model and Configuration
# ///////////////////////////////////////////////////////
print("Saving final model")
torch.save(m.state_dict(), f'{args.output}/result.pt')
with open(f'{args.output}/config.json', 'w') as json_file:
    json.dump(config_data, json_file, indent=4)
