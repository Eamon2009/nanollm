from torch.nn import functional as F
import torch.nn as nn
import torch


torch.seed(1337)
vocab_size = 21
chars2 = ['e', 'a', 'm', 'o', 'n', 'b', 'c', 'd']
stri = {}
for i in range(len(chars2)):
    ch = chars2[i]
    stri[ch] = i

it = {}
for i in range(len(chars2)):
    ch = chars2[i]
    it[i] = ch


def encode(s):
    encoded_list = []
    for c in s:
        encoded_list.append(stri[c])
    return encoded_list


def decode(l):
    decoded_chars = []
    for i in l:
        decoded_chars.append(it[i])
    return "".join(decoded_chars)


torch.manual_seed(1337)
batch_size = 4
block_size = 8


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


xb, yb = get_batch('train')
# -------------------------------------------


class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = torch.view(B*T, C)
            targets = torch.view(B*T)
            loss = F.cross_entropy(logits, targets)

            return loss, logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


m = LanguageModel(vocab_size)
