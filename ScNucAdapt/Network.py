import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_dim, latent_dim,other_dim,classes):  # 2D latent for direct plotting
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, other_dim),
            nn.ReLU(),

            nn.Linear(other_dim, latent_dim)
        )

        self.classifier =nn.Sequential(
            nn.Linear(latent_dim, other_dim),
            nn.ReLU(),
            nn.Linear(32, classes)
        )
    def forward(self, x):
        z = self.encoder(x)
        classify = self.classifier(z)

        return z,classify