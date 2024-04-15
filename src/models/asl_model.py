import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

class ASLModel(nn.Module):
    def __init__(self, num_classes=28): # 26 alphabets in ASL language + 1 blank category
        super().__init__() #This line calls the constructor of the parent class (nn.Module) to initialize the ASLCNN class. This is necessary to properly set up the model.

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
        
        self.fc1 = nn.Linear(32 * 61 * 61, 128)

        self.fc2 = nn.Linear(128, num_classes)  # 26 classes for ASL alphabet


    def forward(self, x):
        """
        When you create a custom neural network model in PyTorch by subclassing nn.Module, you need to define the forward method. The forward method specifies how the input data should be transformed as it passes through the layers of your mode.
        """

        """
        Applying self.pool twice:
        This is a common practice in convolutional neural networks (CNNs) to reduce the spatial dimensions of the feature maps while retaining important information. The purpose of applying pooling is typically twofold:

        Spatial Downsampling: Pooling reduces the spatial dimensions (width and height) of the feature maps, which helps in reducing the computational complexity of the network and controlling overfitting.

        Feature Invariance: Pooling helps in creating features that are invariant to small translations in the input. This means that even if the input image is shifted slightly, the features extracted by the network remain the same, improving the network's ability to generalize.

        By applying pooling twice, the network is able to progressively reduce the spatial dimensions of the feature maps, capturing increasingly abstract and higher-level features from the input image. This can lead to better performance in tasks such as image classification.


        - self.conv1(x): This calls the __call__ method of self.conv1, which is an instance of nn.Conv2d. This method applies the convolution operation defined by self.conv1 to the input x. 
        - F.relu(...): Applies the ReLU (Rectified Linear Unit) activation function to the output of self.conv1(x). This sets all negative values to zero, introducing non-linearity to the network.
        - self.pool(...): Applies max pooling to the output of the ReLU activation. Max pooling reduces the spatial dimensions of the input by taking the maximum value in each window of the specified size.
        - The result of this line is the output of the first convolutional layer followed by ReLU activation and max pooling.

        """
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) # 
        print("DEBUG 1: ", x.size())  # Print the size of x after the last max pooling operation

        """
         Flattens the output of the convolutional layers and pooling layers into a 1-dimensional tensor, excluding the batch dimension. This prepares the data for the fully connected layers.
        """
        """
        Flattening: The torch.flatten function is used to flatten the input tensor x before passing it through the fully connected layers. This is necessary because fully connected layers expect a 1D input, while the output of convolutional layers is typically a 3D tensor (batch_size x channels x height x width). By flattening the tensor, we convert it into a 1D tensor, which can be passed to the fully connected layers.

        ReLU Activation: The F.relu function is the Rectified Linear Unit (ReLU) activation function. It introduces non-linearity to the network, allowing it to learn complex patterns in the data. ReLU is applied after each fully connected layer to introduce non-linearity to the network's output.

        By applying torch.flatten and F.relu twice, the network processes the input through each fully connected layer, introducing non-linearity and preparing the data for the final output layer (self.fc3). Each layer learns to extract and transform features from the input data, contributing to the overall task of the network, which in this case is ASL sign prediction and caption display.

        """
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        """
        x = F.relu(self.fc1(x)): This line computes the output of the first fully connected layer (self.fc1) and applies the ReLU activation function to introduce non-linearity to the network. The result is stored back in x.

        x = self.fc2(x): This line computes the output of the second (and final) fully connected layer (self.fc2) without applying another activation function like ReLU. This is because the final output of the network does not necessarily need to be passed through an activation function, depending on the task and the network architecture.
        """
        print("DEBUG 2: ", x.size())  # Print the size of x after the last max pooling operation
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x