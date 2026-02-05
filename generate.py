import argparse
import json
import os
import sys
import torch
from languageModel.bigramLanguageModel import BigramLanguageModel
from tokenizer.tokenizer import Tokenizer

print("----- CornellNano-172 Text Generation -----")
# Parse argument for directory holding the GPT model
parser = argparse.ArgumentParser(description="Generate text using a trained GPT model")
parser.add_argument("--gpt_path", type=str, default='savedResults\\40000_iterations', help="directory path to the trained GPT model")
parser.add_argument('--tokens', type=int, default=500, help='Number of tokens to generate')
parser.add_argument('--prompt', type=str, default='', help='Seed text to start generation (optional)')
args = parser.parse_args()

# Check that the GPT directory exists
if os.path.isdir(args.gpt_path):
    with open(f'{args.gpt_path}/config.json', 'r') as configuration:
        config_data = json.load(configuration)
else:
    print(f'ERROR: Folder at folder path {args.gpt_path} does not exist')
    sys.exit()

# Parameters From Config File
block_size = config_data['block_size'] # Maximum context length
n_embd = config_data['n_embd'] # Number of embedding dimensions to use for embeddings
n_layer = config_data['n_layer'] # Number of transformers used in the language model
n_heads = config_data['n_heads'] # Number of heads in each multi-headed attention block
dropout = config_data['dropout'] # Dropout percentage to maintain evolution
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Device to run the language model on

# Load the tokenizer vocabulary that was saved during training
vocab_path = f'{args.gpt_path}/vocab.json'
tokenizer = Tokenizer.load_vocab(vocab_path)

model = BigramLanguageModel(block_size, device, dropout, n_embd, n_heads, n_layer, tokenizer.vocab_size())
model.load_state_dict(torch.load(f'{args.gpt_path}/result.pt', weights_only=True))
model.eval()
m = model.to(device)

# Create input vector representing input token index
if args.prompt:
    # Encode the prompt and use it as seed text
    prompt_tokens = tokenizer.encode(args.prompt)
    context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    print(args.prompt, end='', flush=True)
else:
    # Start with empty context
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Generate new tokens in chunks
tokens_generated = 0
chunk_size = 20

while tokens_generated < args.tokens:
    tokens_to_generate = min(chunk_size, args.tokens - tokens_generated)
    context = m.generate(idx=context, max_new_tokens=tokens_to_generate)
    new_tokens = context[0, -tokens_to_generate:].tolist()
    print(tokenizer.decode(new_tokens), end='', flush=True)
    tokens_generated += tokens_to_generate

print("\n\n----- Generation Complete -----")
