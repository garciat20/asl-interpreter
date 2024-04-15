import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

class ASLModel(nn.Module):
    def __init__(self, num_classes=28): # 26 alphabets in ASL language + 1 blank category
        super().__init__() #This line calls the constructor of the parent class (nn.Module) to initialize the ASLCNN class. This is necessary to properly set up the model.

         # Convolutional layer 1: 3 input channels (RGB), 16 output channels, 3x3 kernel size
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Max pooling layer 1: 2x2 kernel size, stride of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        "change to 128"
        self.fc1 = nn.Linear(32 * 61 * 61, 52) 

        "change to 128"
        self.fc2 = nn.Linear(52, num_classes)  # 26 classes for ASL alphabet


    def forward(self, x):
   
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) # 
        # print("DEBUG 1: ", x.size())  # Print the size of x after the last max pooling operation

        x = torch.flatten(x, 1) # flatten all dimensions except batch

        # print("DEBUG 2: ", x.size())  # Print the size of x after the last max pooling operation
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x