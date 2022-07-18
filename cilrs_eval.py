import os

import yaml

from carla_env.env import Env
import numpy as np
import torch
from torchvision import transforms
class Evaluator():
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.agent = self.load_agent()

    def load_agent(self):
        # Load model from CILRS checkpoint
        model = torch.load("cilrs_model2022-07-18-01-02-53.ckpt")
        model.eval()
        model.to('cpu')
        return model


    def generate_action(self, rgb, command, speed):
        """Generate action from model"""
        # Convert to tensor

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda x: x/255),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        rgb = transform(torch.from_numpy(np.transpose(rgb, (2, 0, 1)))).unsqueeze(0)
        command = torch.tensor(command).float().reshape((1, 1))
        speed = torch.tensor(speed).float().reshape(1, 1)
        # Predict
        _, action = self.agent(rgb, speed, command)
        # Convert to numpy
        steer, acceleration = action.detach().numpy().squeeze()
        # Generate action
        throttle = (acceleration >= 0) * acceleration
        brake = (acceleration < 0) * -acceleration

        return float(throttle), float(steer), float(brake)

    def take_step(self, state):
        rgb = state["rgb"]
        command = state["command"]
        speed = state["speed"]
        throttle, steer, brake = self.generate_action(rgb, command, speed)
        action = {
            "throttle": throttle,
            "brake": brake,
            "steer": steer
        }
        state, reward_dict, is_terminal = self.env.step(action)
        return state, is_terminal

    def evaluate(self, num_trials=100):
        terminal_histogram = {}
        for i in range(num_trials):
            state, _, is_terminal = self.env.reset()
            for i in range(5000):
                if is_terminal:
                    break
                state, is_terminal = self.take_step(state)
            if not is_terminal:
                is_terminal = ["timeout"]
            terminal_histogram[is_terminal[0]] = (terminal_histogram.get(is_terminal[0], 0)+1)
        print("Evaluation over. Listing termination causes:")
        for key, val in terminal_histogram.items():
            print(f"{key}: {val}/100")


def main():
    with open(os.path.join("configs", "cilrs.yaml"), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        evaluator = Evaluator(env, config)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
