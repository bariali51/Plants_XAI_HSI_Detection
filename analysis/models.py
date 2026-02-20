# classifier/models.py
import torch
import torch.nn as nn


class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(PlantDiseaseModel, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.global_avg_pool(x)
        x = self.fc_block(x)
        return x