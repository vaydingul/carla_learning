import torch.nn as nn
import torch
activation_function_map = {"relu": nn.ReLU,
                           "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class VGG16(nn.Module):
    """Import pretrained VGG-16 model from torch"""

    def __init__(self, vgg_type) -> None:
        super().__init__()
        self.model = torch.hub.load(
            'pytorch/vision:v0.10.0', vgg_type, pretrained=True)

    def forward(self, x):
        return self.model(x)


class Branch(nn.Module):
    """A command-based branch switching network"""

    def __init__(self, input_dimension, output_dimension, hidden_dimensions, num_branches, hidden_activation, final_activation):
        """Create different branches of encoders based on the num_branches"""
        super(Branch, self).__init__()

        self.num_branches = num_branches
        self.branches = nn.ModuleList([Encoder(input_dimension, output_dimension, hidden_dimensions,
                                      hidden_activation, final_activation) for _ in range(num_branches)])
        self.stack = Stack()

    def forward(self, x, command):

        #! Figure out a smarter way to do it! Multiply with a mask.
        #x_ = self.stack(*[branch(x) for branch in self.branches])
        #x = torch.index_select(x_, 2, command)

        output = []
        for (ix, c) in enumerate(command):
            output.append(self.branches[int(c)](x[ix]))
        x = self.stack(*output, dim=0)
        return x


class Encoder(nn.Module):
    """An feed-forward network encoder which consists of a sequence of linear layers"""

    def __init__(self, input_dimension, output_dimension, hidden_dimensions, hidden_activation, final_activation):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList(
            [nn.Linear(input_dimension, hidden_dimensions[0])])
        self.layers.extend([nn.Linear(hidden_dimensions[k], hidden_dimensions[k + 1])
                           for k in range(len(hidden_dimensions)-1)])
        self.layers.append(nn.Linear(hidden_dimensions[-1], output_dimension))
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

    def forward(self, *vecs, dim=1):
        return torch.cat((vecs), dim=dim)


class Stack(nn.Module):
    """Concatenate two tensors"""

    def __init__(self):
        super(Stack, self).__init__()

    def forward(self, *vecs, dim=2):
        return torch.stack((vecs), dim=dim)


class TaskBlock(nn.Module):
    """Conditional and unconditional task block implementation"""

    def __init__(self, input_dimension, output_dimension, hidden_dimensions, num_branches, hidden_activation, final_activation, conditional: bool = True) -> None:
        super().__init__()

        self.conditional = conditional

        if conditional:

            self.task_block = Branch(input_dimension, output_dimension,
                                     hidden_dimensions, num_branches, hidden_activation, final_activation)

        else:

            self.task_block = Encoder(input_dimension, output_dimension,
                                     hidden_dimensions, hidden_activation, final_activation)

    def forward(self, x, command):

        if self.conditional:

            return self.task_block(x, command)

        else:

            return self.task_block(x)


class AffordancePredictorLoss(nn.Module):
    """Loss function for the affordance predictor"""

    def __init__(self, weight = 0.7):
        super(AffordancePredictorLoss, self).__init__()
        self.weight = weight
        self.loss1 = nn.L1Loss()
        self.loss2 = nn.BCELoss()

    def forward(self, lane_distance_pred, lane_distance_gt, route_angle_pred, route_angle_gt, traffic_light_distance_pred, traffic_light_distance_gt, traffic_light_state_pred, traffic_light_state_gt):
        loss1 = self.loss1(lane_distance_pred, lane_distance_gt)
        loss2 = self.loss1(route_angle_pred, route_angle_gt)
        loss3 = self.loss1(traffic_light_distance_pred,
                           traffic_light_distance_gt)
        loss4 = self.loss2(traffic_light_state_pred, traffic_light_state_gt)

        #return self.weight * (loss1 + loss2 + loss3) + (1 - self.weight) * loss4
        return loss1 + loss2 + loss3 + loss4

class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""

    def __init__(self, config):
        super(AffordancePredictor, self).__init__()
        self.image_encoder = VGG16(config["image_encoder_type"])
        self.lane_distance_task_block = TaskBlock(config["lane_distance_encoder_input_dimension"], config["lane_distance_encoder_output_dimension"], config["lane_distance_encoder_hidden_layer_dimensions"], config["number_of_commands"],
                                                  activation_function_map[config["lane_distance_encoder_hidden_activation"]], activation_function_map[config["lane_distance_encoder_final_activation"]], conditional=True)
        self.route_angle_task_block = TaskBlock(config["lane_angle_encoder_input_dimension"], config["lane_angle_encoder_output_dimension"], config["lane_angle_encoder_hidden_layer_dimensions"], config["number_of_commands"],
                                                activation_function_map[config["lane_angle_encoder_hidden_activation"]], activation_function_map[config["lane_angle_encoder_final_activation"]], conditional=True)
        self.traffic_light_distance_task_block = TaskBlock(config["traffic_light_distance_encoder_input_dimension"], config["traffic_light_distance_encoder_output_dimension"], config["traffic_light_distance_encoder_hidden_layer_dimensions"], config["number_of_commands"],
                                                           activation_function_map[config["traffic_light_distance_encoder_hidden_activation"]], activation_function_map[config["traffic_light_distance_encoder_final_activation"]], conditional=False)
        self.traffic_light_state_task_block = TaskBlock(config["traffic_light_state_encoder_input_dimension"], config["traffic_light_state_encoder_output_dimension"], config["traffic_light_state_encoder_hidden_layer_dimensions"], config["number_of_commands"],
                                                        activation_function_map[config["traffic_light_state_encoder_hidden_activation"]], activation_function_map[config["traffic_light_state_encoder_final_activation"]], conditional=False)
        self.loss_criterion = AffordancePredictorLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=config["learning_rate"])

    def forward(self, img, command):

        x = self.image_encoder(img)

        lane_distance = self.lane_distance_task_block(x, command)
        route_angle = self.route_angle_task_block(x, command)
        traffic_light_distance = self.traffic_light_distance_task_block(x, command)
        traffic_light_state = self.traffic_light_state_task_block(x, command)

        return lane_distance, route_angle, traffic_light_distance, traffic_light_state
