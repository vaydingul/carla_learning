from enum import Enum
from git import Object
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import json
import torch
from torchvision import transforms
import h5py

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
        h5_files = sorted(os.listdir(self.data_root))
        images = []
        command = []
        speed = []
        throttle = []
        steer = []
        brake = []


        for h5_file in h5_files:
            
            print("Loading {}".format(h5_file))
            
            f = h5py.File(os.path.join(self.data_root, h5_file), 'r')
       
            for k in range(len(list(f.keys()))):
                
                images.append(f[f"step_{k}"]["obs"]["central_rgb"]["data"])
                command.append(f[f"step_{k}"]["obs"]["gnss"]["command"][0])
                speed.append(f[f"step_{k}"]["obs"]["speed"]["speed"][0])
                throttle.append(f[f"step_{k}"]["supervision"]["action"][0])
                steer.append(f[f"step_{k}"]["supervision"]["action"][1])
                brake.append(f[f"step_{k}"]["supervision"]["action"][2])
            #f.close()
        #self._images = torch.stack([torch.from_numpy(np.transpose(im[:], (2, 0, 1))) for im in images], dim = 0)
        self._images = images
        self._command = torch.from_numpy(np.array(command)).unsqueeze(1)
        self._speed = torch.from_numpy(np.array(speed)).unsqueeze(1)
        self._steer = torch.from_numpy(np.array(steer)).unsqueeze(1)
        self._throttle = torch.from_numpy(np.array(throttle)).unsqueeze(1)
        self._brake = torch.from_numpy(np.array(brake)).unsqueeze(1)
        self._length = len(self._images)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        """Return RGB images and measurements"""

        img = self.transform(torch.Tensor(np.transpose(self._images[index][:], (2, 0, 1))))

        if self.learning_type == LearningType.IMITATION:

            return img, self._command[index], self._speed[index], self._steer[index], self._throttle[index], self._brake[index]

        elif self.learning_type == LearningType.AFFORDANCE:

            NotImplementedError(
                "Data acquisition for affordance learning is not implemented yet")


        elif self.learning_type == LearningType.REINFORCEMENT:

            NotImplementedError(
                "Data acquisition for reinforcement learning is not implemented yet")

        else:

            pass

    def __len__(self):

        return self._length


if __name__ == "__main__":

    expert_dataset = ExpertDataset(
        "/home/vaydingul20/Documents/Codes/carla_learning/subsubset/expert",
         learning_type=LearningType.IMITATION)

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
