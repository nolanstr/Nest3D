import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, n):
        super(Discriminator, self).__init__()
        
        # Define the layers of the discriminator
        self.layer1 = nn.Linear(n * n, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, x):
        # Flatten the input
        x = x.flatten()
        
        # Pass the input through the layers with ReLU activations
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        
        # The final layer uses a sigmoid activation to output a probability
        x = torch.sigmoid(self.layer3(x))
        
        return x
