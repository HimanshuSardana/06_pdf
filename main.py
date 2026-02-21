import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

seed_val = 102303244
alpha = 0.5 * (seed_val % 7)
beta = 0.3 * (seed_val % 5 + 1)

data = pd.read_csv("data.csv", encoding="latin1", low_memory=False)
feature = data["no2"].dropna().values.reshape(-1, 1)
target = (feature + alpha) * np.sin(beta * feature)

scaler = RobustScaler()
target_norm = scaler.fit_transform(target)
target_tensor = torch.tensor(target_norm, dtype=torch.float32)


class GenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, input_tensor):
        return self.layers(input_tensor)


class DiscModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_tensor):
        return self.layers(input_tensor)


generator = GenModel()
discriminator = DiscModel()
loss_fn = nn.BCELoss()
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0005)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0005)

num_epochs = 2000
batch_sz = 64
num_samples = target_tensor.shape[0]

for epoch in range(num_epochs):
    indices = torch.randint(0, num_samples, (batch_sz,))
    real_samples = target_tensor[indices]

    noise = torch.randn(batch_sz, 1)
    generated_samples = generator(noise)

    real_targets = torch.ones(batch_sz, 1)
    fake_targets = torch.zeros(batch_sz, 1)

    disc_real = discriminator(real_samples)
    disc_fake = discriminator(generated_samples.detach())

    disc_loss = loss_fn(disc_real, real_targets) + loss_fn(disc_fake, fake_targets)

    disc_optimizer.zero_grad()
    disc_loss.backward()
    disc_optimizer.step()

    noise = torch.randn(batch_sz, 1)
    generated_samples = generator(noise)
    disc_fake = discriminator(generated_samples)

    gen_loss = loss_fn(disc_fake, real_targets)

    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()

    if epoch % 500 == 0:
        print(
            f"Epoch {epoch}: Discriminator Loss: {disc_loss.item():.4f} | Generator Loss: {gen_loss.item():.4f}"
        )

with torch.no_grad():
    test_noise = torch.randn(10000, 1)
    synthetic_data = generator(test_noise).numpy()

synthetic_data = scaler.inverse_transform(synthetic_data)

plt.hist(synthetic_data, bins=100, density=True, alpha=0.7)
plt.xlabel("Transformed Variable")
plt.ylabel("Estimated Density")
plt.title("PDF Estimated by GAN")
plt.show()
plt.savefig("histogram.png")

plt.hist(target, bins=100, density=True, alpha=0.4, label="Original Data")
plt.hist(synthetic_data, bins=100, density=True, alpha=0.8, label="Generated Data")
plt.legend()
plt.savefig("gan_comparison.png")
