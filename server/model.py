import io

import torch
from torchvision import transforms
from torch import nn
from PIL import Image


class SmileClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.5),
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.5),
            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=128, out_channels=256,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.classifier(x)
    

def tranform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])

    image = Image.open(io.BytesIO(image_bytes))

    return transform(image).unsqueeze(0)


