import torch
batch_size = 20   # These many independent sequences will we process in parallel
block_size = 128  # This is the maximum context length for predictions
# ------------


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
sttoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(l):
    encoded = []
    for i in l:
        encoded.append(sttoi[i])
    return encoded


def decode(l): return ''.join([itos[i] for i in l])


# Converting into tensor
data = torch.tensor(encode(text), dtype=torch.long)
n = (0.9*len(data))    # Data split
train_data = data[:n]  # 90% training Data
val_data = data[n:]    # 10% validation Data
