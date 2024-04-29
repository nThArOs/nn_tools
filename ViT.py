import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt

np.random.seed(123)
torch.manual_seed(123)

n_patches = 7
n_heads = 2
n_block = 1
hidden_d = 8
out_d = 10
batch_size = 64
# HyperParam
Epoch = 1
lr = 0.005


class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.MultiHeadAttentionAttention = MultiHeadSelfAttention(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.MultiHeadAttentionAttention(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class ViT(nn.Module):
    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        super(ViT, self).__init__()

        # Attributes
        self.chw = chw  # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        assert chw[1] % n_patches == 0, "Input shape not entierly divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entierly divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1 Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2 Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3 Positional Embedding
        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(n_patches ** 2 + 1, hidden_d),
            persistent=False,
        )

        # 4 Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )

        # 5 Classification MLPK
        self.mlp = nn.Sequential(nn.Linear(self.hidden_d, out_d), nn.Softmax(dim=-1))

    def forward(self, images):
        # Dividing image into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # Running linear tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        

        #fig, axs = plt.subplots(2)
        #axs[0].imshow(tokens[0, :, :].detach())
        #axs[1].imshow(out[0, :, :].detach())
        #plt.show()
        # Transformer Blocks
        for block in self.blocks:

            out = block(out)

        # Getting the classification token only
        out = out[:, 0]

        return self.mlp(out)

        return tokens


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MultiHeadSelfAttention, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"d must be divisible by n_heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)     (through concatenation)
        result = []

        for sequence in sequences:
            seq_result = []

            for head in range(self.n_heads):

                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)
                #fig, axs = plt.subplots(3)
                #axs[0].imshow(q.detach())
                #axs[1].imshow(k.detach())
                #axs[2].imshow(v.detach())
                #plt.show()
                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                #print(f"q_k = {q_mapping},k_k = {k_mapping},v_k = {v_mapping}")
                #print(f"attention {attention.shape}")
                #print(f"d = {self.d}")
                #print(f"q = {q.shape},k = {k.shape},v = {v.shape}")
                seq_result.append(attention @ v)
                #print(f"attention heads {(attention @ v).shape}")

            result.append(torch.hstack(seq_result))

        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square image only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patche_size = h // n_patches

    cols = 7
    rows = 7
    #fig, axs = plt.subplots(rows,cols)

    for idx, images in enumerate(images):
        # plt.imshow(images[0,:,:])
        # plt.show()
        for i in range(n_patches):
            for j in range(n_patches):
                patch = images[:, i * patche_size: (i + 1) * patche_size, j * patche_size: (j + 1) * patche_size]
                patches[idx, i * n_patches + j] = patch.flatten()
                #axs[i,j].imshow(patch[0,:,:])
                #axs[i,j].label_outer()

        # plt.figure()
        # plt.imshow(images[0,:,:])
        # plt.figure()
        # plt.imshow(patches[0,:,:])
        #plt.show()
    return patches


def get_positional_embeddings(sequence_length, d):
    #print(sequence_length, d)
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    # plt.imshow(result, interpolation="nearest")
    # plt.show()
    return result


def main():
    # Load Data
    transform = ToTensor()

    train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
    test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViT((1, 28, 28), n_patches=n_patches, n_blocks=n_block, hidden_d=hidden_d, n_heads=n_heads, out_d=out_d).to(
        device)

    otpimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    for epoch in trange(Epoch, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            loss = criterion(y_hat, y)
            #plt.imshow(y_hat[1].detach())


            plt.show()
            train_loss += loss.detach().cpu().item() / len(train_loader)

            otpimizer.zero_grad()
            loss.backward()
            otpimizer.step()

        print(f"Epoch {epoch + 1} /{Epoch} Loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            #print(f" y_hat = {torch.argmax(y_hat, dim=1)}")
            #print(f"y = {y}")

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test Acc : {correct / total * 100:.2f}%")


if __name__ == '__main__':
    # plt.imshow(get_positional_embeddings(100, 300), cmap="hot", interpolation="nearest")
    # plt.show()
    main()
