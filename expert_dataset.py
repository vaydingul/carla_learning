from enum import Enum
from git import Object
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import json
import torch
from torchvision import transforms


class LearningType(Enum):

    """
    Enum for learning type.
    """

    IMITATION = 0
    AFFORDANCE = 1
    REINFORCEMENT = 2


class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""

    def __init__(self, data_root, learning_type=LearningType.IMITATION):
        self.data_root = data_root

        # 0 = imitation learning
        # 1 = direct perception (affordance learning)
        # 2 = reinforcement learning
        self.learning_type = learning_type

        # Fetch the folder content
        rgb_files = sorted(os.listdir(os.path.join(self.data_root, "rgb")))
        measurements_actions = sorted(os.listdir(
            os.path.join(self.data_root, "measurements")))
        self._length = len(rgb_files)

        self._images = torch.zeros(
            (self._length, 3, 512, 512), dtype=torch.float32)
        self._steer = torch.zeros((self._length, 1), dtype=torch.float32)
        self._throttle = torch.zeros((self._length, 1), dtype=torch.float32)
        self._brake = torch.zeros((self._length, 1), dtype=torch.float32)
        self._speed = torch.zeros((self._length, 1), dtype=torch.float32)
        self._command = torch.zeros((self._length, 1), dtype=torch.float32)
        self._route_dist = torch.zeros((self._length, 1), dtype=torch.float32)
        self._route_angle = torch.zeros((self._length, 1), dtype=torch.float32)
        self._lane_dist = torch.zeros((self._length, 1), dtype=torch.float32)
        self._lane_angle = torch.zeros((self._length, 1), dtype=torch.float32)
        self._hazard = torch.zeros((self._length, 1), dtype=torch.float32)
        self._hazard_dist = torch.zeros((self._length, 1), dtype=torch.float32)
        self._tl_state = torch.zeros((self._length, 1), dtype=torch.float32)
        self._tl_dist = torch.zeros((self._length, 1), dtype=torch.float32)
        self._is_junction = torch.zeros((self._length, 1), dtype=torch.float32)

        for k in range(len(rgb_files)):
            # Read images as a NumPy array

            self._images[k] = torch.permute(torch.from_numpy(np.array(Image.open(
                os.path.join(self.data_root, "rgb", rgb_files[k]))))[:, :, :3], (2, 0, 1))
            # Read steer angle command from json file
            with open(os.path.join(self.data_root, "measurements", measurements_actions[k]), "r") as f:
                json_content = json.load(f)
                self._steer[k] = json_content["steer"]
                self._throttle[k] = json_content["throttle"]
                self._brake[k] = json_content["brake"]
                self._speed[k] = json_content["speed"]
                self._command[k] = json_content["command"]
                self._route_dist[k] = json_content["route_dist"]
                self._route_angle[k] = json_content["route_angle"]
                self._lane_dist[k] = json_content["lane_dist"]
                self._lane_angle[k] = json_content["lane_angle"]
                self._hazard[k] = json_content["hazard"]
                self._hazard_dist[k] = json_content["hazard_dist"]
                self._tl_state[k] = json_content["tl_state"]
                self._tl_dist[k] = json_content["tl_dist"]
                self._is_junction[k] = json_content["is_junction"]

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        """Return RGB images and measurements"""

        if self.learning_type == LearningType.IMITATION:

            return self._images[index], self._command[index], self._speed[index], self._steer[index], self._throttle[index], self._brake[index]

        elif self.learning_type == LearningType.AFFORDANCE:

            return self._images[index], self._command[index], self._lane_dist[index], self._lane_angle[index], self._tl_dist[index], self._tl_state[index]

        elif self.learning_type == LearningType.REINFORCEMENT:

            NotImplementedError(
                "Data acquisition for reinforcement learning is not implemented yet")

        else:

            pass

    def __len__(self):

        return self._length


if __name__ == "__main__":

    expert_dataset = ExpertDataset(
        "./dataset/", learning_type=LearningType.IMITATION)

    train_loader = DataLoader(expert_dataset, batch_size=1, shuffle=True,
                              drop_last=True)

    print(expert_dataset[0])
    for i, (images, command, speed, steer, throttle, brake) in enumerate(train_loader):
        print(images.shape)
        print(command.shape)
        print(speed.shape)
        print(steer.shape)
        print(throttle.shape)
        print(brake.shape)
        break
