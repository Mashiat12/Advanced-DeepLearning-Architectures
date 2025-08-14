# Mashiat Tabassum Khan
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(111)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_set = torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transform
)
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

real_samples, mnist_labels = next(iter(train_loader))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])
plt.show()

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)  # Flatten input images
        output = self.model(x)
        return output

discriminator = Discriminator().to(device=device)

# Define the Generator model
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)  # Reshape to (batch_size, 1, 28, 28)
        return output

generator = Generator().to(device=device)

# Hyperparameters
lr = 0.0002
num_epochs = 2000

loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

# Training Loop
for epoch in range(num_epochs):
    for n, (real_samples, mnist_labels) in enumerate(train_loader):
        real_samples = real_samples.to(device=device)
        batch_size_actual = real_samples.shape[0]  # Handle last batch size change

        # FIX 1: Correct label creation using actual batch size
        real_samples_labels = torch.ones((batch_size_actual, 1)).to(device=device)
        latent_space_samples = torch.randn((batch_size_actual, 100)).to(device=device)
        generated_samples = generator(latent_space_samples)

        generated_samples_labels = torch.zeros((batch_size_actual, 1)).to(device=device)

        # FIX 2: Flatten generated samples before passing to discriminator
        generated_samples = generated_samples.view(batch_size_actual, -1)

        # Concatenate real and generated samples with their labels
        all_samples = torch.cat((real_samples.view(batch_size_actual, -1), generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        # Training the Discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Training the Generator
        latent_space_samples = torch.randn((batch_size_actual, 100)).to(device=device)
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)

        # FIX 3: Flatten before passing to discriminator
        generated_samples = generated_samples.view(batch_size_actual, -1)
        output_discriminator_generated = discriminator(generated_samples)

        # FIX 4: Generator should use ones as labels (not real_samples_labels)
        loss_generator = loss_function(output_discriminator_generated, torch.ones_like(output_discriminator_generated))
        loss_generator.backward()
        optimizer_generator.step()

        # FIX 5: Print loss every 100 batches instead of incorrect condition
        if n % 100 == 0:
            print(f"Epoch: {epoch} | Batch: {n} | Loss D.: {loss_discriminator:.4f} | Loss G.: {loss_generator:.4f}")


# Generate images after training
import matplotlib.pyplot as plt

latent_space_samples = torch.randn(16, 100).to(device)
generated_samples = generator(latent_space_samples).cpu().detach()

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated_samples[i].squeeze(), cmap="gray_r")
    ax.axis("off")
plt.show()
