"""Lab 2: Image synthesis using a basic GAN on Fashion-MNIST.

Follows the lab manual steps 1 to 4. Run from this folder:  python lab2_gan.py
"""
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = Path("lab_outputs")
out.mkdir(exist_ok=True)
print("Device:", device)

# step 1: load and scale to [-1, 1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
DATA = str(Path(__file__).resolve().parent.parent / "data")
data = datasets.FashionMNIST(DATA, train=True, download=True, transform=transform)
loader = DataLoader(data, batch_size=128, shuffle=True)

# step 2: networks
latent_dim = 64
G = nn.Sequential(nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
                  nn.Linear(256, 512), nn.LeakyReLU(0.2),
                  nn.Linear(512, 784), nn.Tanh()).to(device)
D = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.LeakyReLU(0.2),
                  nn.Linear(512, 256), nn.LeakyReLU(0.2),
                  nn.Linear(256, 1)).to(device)
loss_fn = nn.BCEWithLogitsLoss()
opt_g = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
fixed_z = torch.randn(25, latent_dim, device=device)

# shape and range checkpoints
real0, _ = next(iter(loader))
with torch.no_grad():
    f0 = G(torch.randn(real0.size(0), latent_dim, device=device))
    d0 = D(real0.to(device))
print("real range:", round(real0.min().item(), 2), round(real0.max().item(), 2),
      "| G out:", tuple(f0.shape), round(f0.min().item(), 2), round(f0.max().item(), 2),
      "| D out:", tuple(d0.shape))


# step 3: alternating updates
def generate_grid(epoch):
    with torch.no_grad():
        images = G(fixed_z).cpu().reshape(-1, 1, 28, 28)
    grid = make_grid((images + 1) / 2, nrow=5)
    plt.figure(figsize=(5, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"Epoch {epoch}")
    plt.axis("off")
    plt.savefig(out / f"gan_epoch_{epoch:02d}.png", bbox_inches="tight", dpi=150)
    plt.close()


def train_batch(real):
    n = real.size(0)
    fake = G(torch.randn(n, latent_dim, device=device))
    real_logits = D(real)
    fake_logits = D(fake.detach())
    d_loss = (loss_fn(real_logits, torch.ones_like(real_logits))
              + loss_fn(fake_logits, torch.zeros_like(fake_logits))) / 2
    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()

    D.requires_grad_(False)
    opt_g.zero_grad()
    logits = D(fake)
    g_loss = loss_fn(logits, torch.ones_like(logits))
    g_loss.backward()
    opt_g.step()
    D.requires_grad_(True)
    return d_loss.item(), g_loss.item()


# step 4: train and compare epochs
history = []
for epoch in range(1, 11):
    losses = [train_batch(real.to(device)) for real, _ in loader]
    avg_d, avg_g = [sum(values) / len(losses) for values in zip(*losses)]
    history.append((avg_d, avg_g))
    print(f"{epoch:02d} D={avg_d:.3f} G={avg_g:.3f}")
    if epoch in (1, 5, 10):
        generate_grid(epoch)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), [d for d, g in history], label="Discriminator")
plt.plot(range(1, 11), [g for d, g in history], label="Generator")
plt.xlabel("Epoch")
plt.ylabel("Mean batch loss")
plt.legend()
plt.savefig(out / "gan_losses.png", bbox_inches="tight", dpi=150)
plt.close()
torch.save(G.state_dict(), out / "fashion_generator.pt")

json.dump({"device": str(device), "seed": SEED,
           "epochs": [{"epoch": i + 1, "D": round(d, 4), "G": round(g, 4)} for i, (d, g) in enumerate(history)]},
          open(out / "results.json", "w"), indent=2)
