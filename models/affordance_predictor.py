from turtle import forward
import torch.nn as nn
import torch

class VGG16(nn.Module):
    """Import pretrained VGG-16 model from torch"""
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    
    def forward(self, x):
        return self.model(x)


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



class TaskBlock(nn.Module):
    """Conditional and unconditional task block implementation"""
    def __init__(self, conditional:bool = True) -> None:
        super().__init__()  

        self.conditional = conditional

        if conditional:

            self.task_block = Branch()

        else:

            self.task_block = Encoder()

    def forward(self, x, command):

        if self.conditional:

            return self.task_block(x, command)

        else:

            return self.task_block(x)


class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""
    def __init__(self):
        super(AffordancePredictor, self).__init__()
        self.image_encoder = VGG16()
        self.lane_distance_task_block = TaskBlock(conditional=True)
        self.route_angle_task_block = TaskBlock(conditional=True)
        self.traffic_light_distance_task_block = TaskBlock(conditional=False)
        self.traffic_light_state_task_block = TaskBlock(conditional=False)

    def forward(self, img, command):

        x = self.image_encoder(img)
        
        lane_distance = self.lane_distance_task_block(x, command)
        route_angle = self.route_angle_task_block(x, command)
        traffic_light_distance = self.traffic_light_distance_task_block(x)
        traffic_light_state = self.traffic_light_state_task_block(x)

        return lane_distance, route_angle, traffic_light_distance, traffic_light_state
        
