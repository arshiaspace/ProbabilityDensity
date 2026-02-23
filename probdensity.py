# ============================================
# Learning PDF using GAN — Assignment
# Dataset: India Air Quality (NO2 column)
# Roll Number: 102303144
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch
import torch.nn as nn
import torch.optim as optim

r = 102303144

a_r = 0.5 * (r % 7)
b_r = 0.3 * ((r % 5) + 1)

print("Transformation Parameters")
print("a_r =", a_r)
print("b_r =", b_r)



file_path = r"C:\Users\Arshia\OneDrive\Desktop\ProbabilityDensity\data.csv"

data = pd.read_csv(file_path, encoding="latin1", low_memory=False)

print("\nDataset Loaded Successfully")
print("Columns:", data.columns)

x = data["no2"].dropna().values

print("Total samples:", len(x))


z = x + a_r * np.sin(b_r * x)
z = z.reshape(-1, 1)

z_mean = z.mean()
z_std = z.std()

z_norm = (z - z_mean) / z_std


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_real = torch.tensor(z_norm, dtype=torch.float32).to(device)

latent_dim = 5


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()

opt_G = optim.Adam(G.parameters(), lr=0.001)
opt_D = optim.Adam(D.parameters(), lr=0.001)

epochs = 3000
batch_size = 128

losses_G = []
losses_D = []

for epoch in range(epochs):

    idx = np.random.randint(0, z_real.shape[0], batch_size)
    real_samples = z_real[idx]

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    noise = torch.randn(batch_size, latent_dim).to(device)
    fake_samples = G(noise)

    D_real = D(real_samples)
    D_fake = D(fake_samples.detach())

    loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)

    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    noise = torch.randn(batch_size, latent_dim).to(device)
    fake_samples = G(noise)

    D_fake = D(fake_samples)

    loss_G = criterion(D_fake, real_labels)

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    losses_G.append(loss_G.item())
    losses_D.append(loss_D.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")


with torch.no_grad():
    noise = torch.randn(5000, latent_dim).to(device)
    gen_samples = G(noise).cpu().numpy()

gen_samples = gen_samples * z_std + z_mean
gen_samples = gen_samples.flatten()


kde = gaussian_kde(gen_samples)

x_range = np.linspace(min(gen_samples), max(gen_samples), 500)
pdf = kde(x_range)


plt.figure(figsize=(8,5))
plt.plot(x_range, pdf)
plt.title("Estimated PDF from GAN Samples")
plt.xlabel("z")
plt.ylabel("Density")
plt.grid()
plt.show()


plt.figure(figsize=(8,5))
plt.hist(z.flatten(), bins=50, density=True, alpha=0.5, label="Real")
plt.hist(gen_samples, bins=50, density=True, alpha=0.5, label="Generated")
plt.legend()
plt.title("Real vs Generated Distribution")
plt.show()


plt.figure(figsize=(8,5))
plt.plot(losses_G, label="Generator Loss")
plt.plot(losses_D, label="Discriminator Loss")
plt.legend()
plt.title("Training Loss")
plt.show()