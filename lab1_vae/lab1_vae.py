"""Lab 1: Variational Autoencoder for image generation (MNIST).

Follows the lab manual steps 1 to 4. Extra additions: loss logging to JSON,
a finite-loss check, and a test-set loss so the README can quote real numbers.
Run from this folder:  python lab1_vae.py
"""
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

# ---- common setup cell ----
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = Path("lab_outputs")
out.mkdir(exist_ok=True)
print("Device:", device)

# ---- step 1: load MNIST ----
DATA = str(Path(__file__).resolve().parent.parent / "data")
train = datasets.MNIST(DATA, train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST(DATA, train=False, download=True, transform=transforms.ToTensor())
loader = DataLoader(train, batch_size=128, shuffle=True)


# ---- step 2: build the VAE ----
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.mean = nn.Linear(256, 2)
        self.logvar = nn.Linear(256, 2)
        self.decoder = nn.Sequential(nn.Linear(2, 256), nn.ReLU(),
                                     nn.Linear(256, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        return self.mean(h), self.logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar


vae = VAE().to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

# shape checkpoints from the manual
xb, _ = next(iter(loader))
xb = xb.to(device)
with torch.no_grad():
    rec, mu_, lv_ = vae(xb)
print("Input batch:", tuple(xb.shape), "| mean/logvar:", tuple(mu_.shape), tuple(lv_.shape),
      "| reconstruction:", tuple(rec.shape))
assert tuple(xb.shape[1:]) == (1, 28, 28) and mu_.shape[1] == 2 and rec.shape[1] == 784

# ---- step 3: train and record both loss terms ----
log = []
for epoch in range(1, 6):
    vae.train()
    totals = dict(reconstruction=0.0, kl=0.0)
    for x, _ in loader:
        x = x.to(device)
        generated, mu, logvar = vae(x)
        recon = F.binary_cross_entropy(generated, x.flatten(1), reduction="sum")
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum()
        loss = (recon + kl) / x.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totals["reconstruction"] += recon.item()
        totals["kl"] += kl.item()
    row = {k: round(v / len(train), 2) for k, v in totals.items()}
    row["total"] = round(row["reconstruction"] + row["kl"], 2)
    assert all(math.isfinite(v) for v in row.values()), "non-finite loss"
    log.append({"epoch": epoch, **row})
    print(epoch, row)
torch.save(vae.state_dict(), out / "vae_mnist.pt")

# ---- step 4: sampling and evaluation ----
def show_grid(images, name, nrow=4, size=(5, 5)):
    grid = make_grid(images.cpu(), nrow=nrow)
    plt.figure(figsize=size)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(out / name, bbox_inches="tight", dpi=150)
    plt.close()


vae.eval()
with torch.no_grad():
    x, y = next(iter(DataLoader(test, batch_size=1024)))
    mu, _ = vae.encode(x.to(device))
    recon = vae.decoder(mu[:8]).reshape(-1, 1, 28, 28)
    samples = vae.decoder(torch.randn(16, 2, device=device))
    show_grid(torch.cat([x[:8], recon.cpu()]), "vae_reconstructed.png", nrow=8, size=(8, 2.4))
    show_grid(samples.reshape(-1, 1, 28, 28), "vae_generated.png")
    coords = mu.cpu()

    # extra: loss on the whole held-out set, for the README
    tl = dict(reconstruction=0.0, kl=0.0)
    for xt, _ in DataLoader(test, batch_size=1000):
        g, m, lv = vae(xt.to(device))
        tl["reconstruction"] += F.binary_cross_entropy(g, xt.to(device).flatten(1), reduction="sum").item()
        tl["kl"] += (-0.5 * (1 + lv - m.square() - lv.exp()).sum()).item()
    test_loss = {k: round(v / len(test), 2) for k, v in tl.items()}
    print("held-out loss per image:", test_loss)

plt.figure(figsize=(6, 4))
plt.scatter(coords[:, 0], coords[:, 1], c=y, cmap="tab10", s=9)
plt.colorbar(ticks=range(10), label="Digit label")
plt.xlabel("Latent dimension 1")
plt.ylabel("Latent dimension 2")
plt.savefig(out / "vae_latent.png", bbox_inches="tight", dpi=150)
plt.close()

# per-digit centroid distances, to back up the "which labels overlap" comment with numbers
cent = {d: coords[y == d].mean(0) for d in range(10)}
pairs = sorted(((float((cent[a] - cent[b]).norm()), a, b) for a in range(10) for b in range(a + 1, 10)))
closest = [(a, b, round(dist, 3)) for dist, a, b in pairs[:5]]
print("closest centroid pairs:", closest)

json.dump({"device": str(device), "seed": SEED, "epochs": log,
           "heldout_loss_per_image": test_loss, "closest_centroids": closest},
          open(out / "results.json", "w"), indent=2)
