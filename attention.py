import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'mps'
torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
with open("data/tinychen.txt", "r") as f:
    tinychen = f.read()
alphabet = sorted(list(set(tinychen)))
vocab_size=len(alphabet)
itos = {i:s for i,s in enumerate(alphabet)}
stoi = {s:i for i,s in itos.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda t: ''.join([itos[tk] for tk in t])
data = torch.tensor(encode(tinychen), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]
batch_size = 64
block_size = 256
n_embed = 384
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_layer = 6
n_head = 6
dropout = 0.2

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x,y


class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embed, vocab_size) 
        self.ln_f = nn.LayerNorm(n_embed)
    
    def forward(self, idx, targets=None): 
        _,T = idx.shape
        tok_emb = self.embedding_table(idx)
        pos_emb = self.pos_table(torch.arange(T, device=device))
        logits = self.lm_head(self.blocks(tok_emb + pos_emb))
        if targets == None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.shape[2]), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.concat((idx, idx_next), dim=1)
        return idx

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril == 0,float('-inf'))
        wei = F.softmax(wei, -1)
        wei = self.dropout(wei)
        out = wei @ v 
        return out 

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.projection(torch.cat([h(x) for h in self.heads], -1)))

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.ffwd = FeedForward(n_embed)
        self.sa = MultiHeadAttention(n_head, head_size) 
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        return self.ffwd(self.ln2(self.sa(self.ln1(x)) + x)) + x

model = BigramLM()
model = model.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            print(f"Model device: {next(model.parameters()).device}")
            print(f"Input tensor device: {X.device}")
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    print("iter")

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, block_size), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
