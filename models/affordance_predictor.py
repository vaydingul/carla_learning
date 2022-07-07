from turtle import forward
import torch.nn as nn
import torch
activation_function_map = {"relu" : nn.ReLU, "tanh" : nn.Tanh, "sigmoid" : nn.Sigmoid}


class VGG16(nn.Module):
    """Import pretrained VGG-16 model from torch"""
    def __init__(self, vgg_type) -> None:
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', vgg_type, pretrained=True)
    
    def forward(self, x):
        return self.model(x)


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
    def __init__(self, input_size, output_size, hidden_sizes, num_branches, hidden_activation, final_activation, conditional:bool = True) -> None:
        super().__init__()  

        self.conditional = conditional

        if conditional:

            self.task_block = Branch(input_size, output_size, hidden_sizes, num_branches, hidden_activation, final_activation)

        else:

            self.task_block = Branch(input_size, output_size, hidden_sizes, 1, hidden_activation, final_activation)

    def forward(self, x, command):

        if self.conditional:

            return self.task_block(x, command)

        else:

            return self.task_block(x, 0)

class AffordancePredictorLoss(nn.Module):
    """Loss function for the affordance predictor"""
    def __init__(self, weight):
        super(AffordancePredictorLoss, self).__init__()
        self.weight = weight
        self.loss1 = nn.L1Loss()
        self.loss2 = nn.BCELoss()

        
    def forward(self, lane_distance_pred, lane_distance_gt, route_angle_pred, route_angle_gt, traffic_light_distance_pred, traffic_light_distance_gt, traffic_light_state_pred, traffic_light_state_gt):
        loss1 = self.loss1(lane_distance_pred, lane_distance_gt)
        loss2 = self.loss1(route_angle_pred, route_angle_gt)
        loss3 = self.loss1(traffic_light_distance_pred, traffic_light_distance_gt)
        loss4 = self.loss2(traffic_light_state_pred, traffic_light_state_gt)

        return self.weight(loss1 + loss2 + loss3) + (1 - self.weight) * loss4


class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""
    def __init__(self, config):
        super(AffordancePredictor, self).__init__()
        self.image_encoder = VGG16()
        self.lane_distance_task_block = TaskBlock(config["encoder_input_size"], config["encoder_output_size"], config["encoder_hidden_layer_dimensions"], config["number_of_commands"], 
                                        activation_function_map[config["branched_encoder_hidden_activation"]], activation_function_map[config["encoder_final_activation"]], conditional=True)
        self.route_angle_task_block = TaskBlock(config["encoder_input_size"], config["encoder_output_size"], config["encoder_hidden_layer_dimensions"], config["number_of_commands"], 
                                        activation_function_map[config["branched_encoder_hidden_activation"]], activation_function_map[config["encoder_final_activation"]], conditional=True)
        self.traffic_light_distance_task_block = TaskBlock(config["encoder_input_size"], config["encoder_output_size"], config["encoder_hidden_layer_dimensions"], config["number_of_commands"], 
                                        activation_function_map[config["branched_encoder_hidden_activation"]], activation_function_map[config["encoder_final_activation"]], conditional=False)
        self.traffic_light_state_task_block = TaskBlock(config["encoder_input_size"], config["encoder_output_size"], config["encoder_hidden_layer_dimensions"], config["number_of_commands"], 
                                        activation_function_map[config["branched_encoder_hidden_activation"]], activation_function_map[config["encoder_final_activation"]], conditional=False)

    def forward(self, img, command):

        x = self.image_encoder(img)
        
        lane_distance = self.lane_distance_task_block(x, command)
        route_angle = self.route_angle_task_block(x, command)
        traffic_light_distance = self.traffic_light_distance_task_block(x)
        traffic_light_state = self.traffic_light_state_task_block(x)

        return lane_distance, route_angle, traffic_light_distance, traffic_light_state
        
