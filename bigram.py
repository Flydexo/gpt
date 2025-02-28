import torch
import torch.nn as nn
from torch.nn import functional as F

n_embed = 32
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
batch_size = 8
block_size = 4
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200

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
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx, targets=None): 
        _,T = idx.shape
        tok_emb = self.embedding_table(idx)
        pos_emb = self.pos_table(torch.arange(T, device=device))
        logits = self.lm_head(tok_emb + pos_emb)
        if targets == None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.shape[2]), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx, )
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.concat((idx, idx_next), dim=1)
        return idx

model = BigramLM()
m = model.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

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
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
