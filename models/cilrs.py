import torch.nn as nn
import torch.hub as hub
import torch

class ResNet18(nn.Module):
    """Basic implementation of ResNet18 from PyTorch Hub"""
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    """An feed-forward network encoder which consists of a sequence of linear layers"""
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.layers.extend([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)])
        self.layers.extend(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nn.ReLU()(x)
        return x

class Concatenate(nn.Module):
    """Concatenate two tensors"""
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, x1, x2):
        return torch.cat((x1, x2), dim=1)

class Branch(nn.Module):
    """A command-based branch switching network"""
    def __init__(self, input_size, output_size, hidden_size, num_layers, num_branches):
        """Create different branches of encoders based on the num_branches"""
        super(Branch, self).__init__()
        
        self.num_branches = num_branches
        self.branches = nn.ModuleList([self.__class__(input_size, output_size, hidden_size, num_layers) for _ in range(num_branches)])
    
    def forward(self, x, command):
        
        x = self.branches[command](x)

        return x

class CILRS(nn.Module):
    """A CILRS imitation learning agent ."""
    def __init__(self, num_commands):
        super(CILRS, self).__init__()

        self.resnet = ResNet18()
        self.measurement_encoder = Encoder(1, 512, 512, 3)
        self.speed_encoder = Encoder(1, 512, 512, 3)
        self.concatenate = Concatenate()
        self.branched_encoder = Branch(512, 512, 512, 3, num_commands)

    def forward(self, img, speed, command):
        
        x1 = self.resnet(img)
        x2 = self.measurement_encoder(speed)
        latent = self.concatenate(x1, x2)
        speed_head = self.speed_encoder(x1)
        action_head = self.branched_encoder(latent, command)
        return speed_head, action_head
