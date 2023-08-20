import torch
import torch.nn as nn
from torch.nn import functional as F
import time

start = time.time()
# Hyperparameters
batch_size = 32 # How many independent sequences will be process in parallel?
block_size = 8 # What is the maximum context length for predictions?
max_iters = 5000
eval__interval = 500
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
print("The Device", device)
eval_iters = 200
n_embed = 32
# ------

torch.manual_seed(1729)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()


# Here are all the unique characters that occur in the corpus
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

# Creating a mapping from charaters to integers
stoi = {ch:i for i,ch in enumerate(vocab)}
itos = {i:ch for i,ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[c] for c in l)


# Split into train data and val data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# Data Loading
def get_batch(split):
    data = train_data if split=="train" else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,), device=device)
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device=device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device=device)
    return x,y


def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = model(X,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Head module

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False) # Key vector, takes in x, and outputs a key(B,T,C)
        self.query = nn.Linear(n_embed, head_size, bias=False) # Query vector, takes in x, and outputs a query(B,T,C)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # Compute attention scores(affinities between the tokens)
        wei = k @ q.transpose(-2, -1) * C**-0.5 # (B, T, C) @ #(B, C, T) --> (B, T, T), Also 'C**-0.5' is for scaled attention 
        # Mask the future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B,T,T)
        wei = F.softmax(wei, dim=-1)
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v #(B, T, T) @ (B, T, C) --> (B, T, T)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention in parallel"""

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1) # Concatenating the outputs of multiple heads
    

# Super simple Bigram model
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits from the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_heads = MultiHeadAttention(4, n_embed//4) # i.e 4 heads of 8-dimensional self-attention
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    
    def forward(self, idx, targets=None):
        
        # idx and targets are both (B,T) tensor of integers
        tok_emd = self.token_embedding_table(idx) #(B, T, C)
        pos_emd = self.position_embedding_table(idx) #(T, C)
        x = tok_emd + pos_emd # (B, T, C)
        x = self.sa_heads(x) # apply one head of self-attention. (B,T,C)
        logits = self.lm_head(x) #(B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Calculate the cross entropy loss
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            # Apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1) #(B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            # append sampled index to the urunning sequence
            idx = torch.cat((idx, idx_next), dim =1) # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device=device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    
    # Every once in a while evaluate the loss on train ad=nd val sets
    if iter % eval__interval == 0:
        losses = estimate_loss()
        print(f"Step: {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # Sample a batch of data
    xb, yb = get_batch("train")
    
    # evaluate the loss
    logits, loss = m(xb, yb)
    # Zeroing out all the gradients from the previous step
    optimizer.zero_grad(set_to_none=True)
    # Getting the parameters from all the gradients
    loss.backward()
    # Use the gradients to update the parameters
    optimizer.step()


print(decode(m.generate(torch.zeros((1,1), dtype=torch.long).to(device=device), max_new_tokens=600)[0].tolist()))
end = time.time()
print(f"Total time taken: {end-start}")
