import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------

batch_size = 34
block_size = 12
max_iter = 1000
eval_itervals = 1
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available else ('cpu')


with open('input.txt', 'r', encoding='utf-8')as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

enc = {std: i for i, std in enumerate(chars)}
cne = {i: std for i, std in enumerate(chars)}
def encode(s): return [enc[c] for c in s]
def decode(l): return ''.join([cne[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_split(split):
    data = train_data if split == 'train' else val_data
    rx = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack(data[i:i+block_size]for i in rx)
    y = torch.stack(data[i+1:i+block_size+1]for i in rx)
    x, y = x.to(device), y.to(device)
    return x, y

# ---------Neural_Net----------------


class DumbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddingTable = nn.Embedding(vocab_size)

    def forward(self, idx, targets=None):
        # here idx is out inputs and now it becomes (B,T,C)
        logits = self.token_embeddingTable(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = torch.view(B*T, C)
            targets = torch.view(B*T)
            loss = F.cross_entropy(logits, targets)

            return logits, loss


model = DumbModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.sgd(model.parameters(), lr=learning_rate)
