###############################################################################
#
# File: model.py
#
# Author: Isaac Ingram
#
# Purpose: A fusion model combining visual, weight sensor, and temporal
# information to determine whether items were taken (or put back) into the
# Bits 'n Byte cabinet.
#
###############################################################################
import torch
import torch.nn as nn



class ByteFusionModel(nn.Module):
    """
    A fusion model to predict interactions with the Bits 'n Bytes cabinet from
    vision data, weight data, and temporal information.
    """

    def __init__(self):
        super().__init__()

        # Neural network for vision data (bounding boxes, confidence score, predictions)
        self.vision_network = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Neural network for weight sensor data
        self.weight_network = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Late fusion: combine processed features
        self.fusion_network = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 50),
            nn.Softmax(dim=1)
        )


    def forward(self, vision_data, weight_data):
        """
        Process vision and weight data to predict interactions,
        :param vision_data: torch.Tensor: Tensor containing YOLO detection outputs.
        :param weight_data: torch.Tensor: Tensor containing weight sensor readings.
        :return: torch.Tensor: Tensor with probabilities for each possible item/action combination.
        """
        # Extract features
        vision_features = self.vision_network(vision_data)
        weight_features = self.weight_network(weight_data)

        # Concatenate features from each mode of input
        combined = torch.cat((vision_features, weight_features), dim=1)

        # Predict using fusion network
        output_probabilities = self.fusion_network(combined)

        return output_probabilities

