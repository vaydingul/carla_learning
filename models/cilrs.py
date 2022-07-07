import torch.nn as nn
import torch.hub as hub
import torch
activation_function_map = {"relu" : nn.ReLU, "tanh" : nn.Tanh}








class ResNet(nn.Module):
    """Basic implementation of ResNet18 from PyTorch Hub"""
    def __init__(self, resnet_type):
        super(ResNet, self).__init__()
        self.model = hub.load('pytorch/vision:v0.6.0', resnet_type, pretrained=True)

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    """An feed-forward network encoder which consists of a sequence of linear layers"""
    def __init__(self, input_size, output_size, hidden_sizes, hidden_activation, final_activation):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        self.layers.extend([nn.Linear(hidden_sizes[k], hidden_sizes[k + 1]) for k in range(len(hidden_sizes))])
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.hidden_activation = hidden_activation()
        self.final_activation = final_activation()

    def forward(self, x):
        for k in range(len(self.layers) - 1):
            x = self.layers[k](x)
            x = self.hidden_activation(x)
        
        x = self.layers[-1](x)
        x = self.final_activation(x)

        return x

class Concatenate(nn.Module):
    """Concatenate two tensors"""
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, x1, x2):
        return torch.cat((x1, x2), dim=1)

class Branch(nn.Module):
    """A command-based branch switching network"""
    def __init__(self, input_size, output_size, hidden_sizes, num_branches, hidden_activation, final_activation):
        """Create different branches of encoders based on the num_branches"""
        super(Branch, self).__init__()
        
        self.num_branches = num_branches
        self.branches = nn.ModuleList([self.__class__(input_size, output_size, hidden_sizes, hidden_activation, final_activation) for _ in range(num_branches)])
    
    def forward(self, x, command):
        
        x = self.branches[command](x)

        return x

class CILRSLoss(nn.Module):
    """Loss function for CILRS"""
    def __init__(self, weight = 0.7):
        super(CILRSLoss, self).__init__()
        self.weight = weight
        self.speed_loss = nn.L1Loss()
        self.action_loss = nn.MSELoss()

    def forward(self, speed_pred, speed_gt, action_pred, action_gt):
        """L1 loss for speed and L2 loss for action"""
        speed_loss = self.speed_loss(speed_pred, speed_gt)
        action_loss = self.action_loss(action_pred, action_gt)
        return (1-self.weight) * speed_loss + self.weight * action_loss


class CILRS(nn.Module):
    """A CILRS imitation learning agent ."""
    def __init__(self, config):
        super(CILRS, self).__init__()

        self.resnet = ResNet()
        self.measurement_encoder = Encoder(config["measurement_encoder_input_size"], config["measurement_encoder_output_size"], config["measurement_encoder_hidden_layer_dimensions"], 
                                            activation_function_map[config["measurement_encoder_hidden_activation"]], activation_function_map[config["measurement_encoder_final_activation"]])
        self.speed_encoder = Encoder(config["speed_encoder_input_size"], config["speed_encoder_output_size"], config["speed_encoder_hidden_layer_dimensions"],
                                        activation_function_map[config["speed_encoder_hidden_activation"]], activation_function_map[config["speed_encoder_final_activation"]])                
        self.concatenate = Concatenate()
        self.branched_encoder = Branch(config["branched_encoder_input_size"], config["branched_encoder_output_size"], config["branched_encoder_hidden_layer_dimensions"], config["number_of_commands"], 
                                        activation_function_map[config["branched_encoder_hidden_activation"]], activation_function_map[config["branched_encoder_final_activation"]])

        self.loss_criterion = CILRSLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["learning_rate"])
        
    def forward(self, img, speed, command):
        
        x1 = self.resnet(img)
        x2 = self.measurement_encoder(speed)
        latent = self.concatenate(x1, x2)
        speed_head = self.speed_encoder(x1)
        action_head = self.branched_encoder(latent, command)
        return speed_head, action_head

