import torch
from torchvision import datasets, transforms
import torch.nn as nn

class ASLModel(nn.Module):
    def __init__(self, num_classes=26): # 26 alphabets in ASL language
        super(ASLModel, self).__init__() #This line calls the constructor of the parent class (nn.Module) to initialize the ASLCNN class. This is necessary to properly set up the model.

         # Define the image preprocessing steps (resize and conversion to tensor)
        # self.preprocessor = transforms.Compose([
        #     transforms.Resize((244, 244)), # a common resizing size for images in machine learning tasks, particularly for convolutional neural networks (CNNs), is 224x224 pixels. 
        #     transforms.ToTensor() # convert images to pytorch tensors | neural networks operate on tensors, so this transformation is necessary to convert the image data into a format that the network can process.
        # ])

         # Convolutional layer 1: 3 input channels (RGB), 16 output channels, 3x3 kernel size
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Max pooling layer 1: 2x2 kernel size, stride of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(32* 122 * 122, 128)

        self.fc2 = nn.Linear(128, out_features=num_classes)  # 26 classes for ASL alphabet

    def forward():
        pass