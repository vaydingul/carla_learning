from enum import Enum
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import json



class LearningType(Enum):

    """
    Enum for learning type.
    """

    IMITATION = 0
    AFFORDANCE = 1
    REINFORCEMENT = 2


class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""
    def __init__(self, data_root, learning_type = LearningType.IMITATION):
        self.data_root = data_root

        # 0 = imitation learning
        # 1 = direct perception (affordance learning)
        # 2 = reinforcement learning
        self.learning_type = learning_type

        # Fetch the folder content
        rgb_files = sorted(os.listdir(os.path.join(self.data_root, "rgb")))
        measurements_actions = sorted(os.listdir(os.path.join(self.data_root, "measurements")))
        self._length = len(rgb_files)

        self._images = []
        self._steer = []
        self._throttle = []
        self._break = []
        self._speed = []
        self._command = []
        self._route_dist = []
        self._route_angle = []
        self._lane_dist = []
        self._lane_angle = []
        self._hazard = []
        self._hazard_dist = []
        self._tl_state = []
        self._tl_dist = []
        self._is_junction = []
        
        for k in range(len(self.rgb_files)):
            # Read images as a NumPy array
            self._images[k] = np.asarray(Image.open(os.path.join(self.data_root, "rgb", rgb_files[k])))
            # Read steer angle command from json file
            with open(os.path.join(self.data_root, "measurements", measurements_actions[k]), "r") as f:
                json_content = json.load(f)
                self._steer[k] = json_content["steer"]
                self._throttle[k] = json_content["throttle"]
                self._break[k] = json_content["break"]
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


        

    def __getitem__(self, index):
        """Return RGB images and measurements"""
    
        if self.learning_type == 0:
            
            return self._images[index], self._command[index], self._speed[index], self._steer[index], self._throttle[index], self._break[index] 

        elif self.learning_type == 1:

            pass

        elif self.learning_type == 2:
            
            pass

        else:

            pass

    def __len__(self):

        return len(self.length)