"""
Steps to use dataset of ASL photos in a pytorch ml model:

1. Organize Your Dataset: Ensure your dataset is organized into a directory structure that PyTorch's ImageDataLoader can use. Have separate folders for each class (e.g., A-Z, 0-9) containing the corresponding images.

2. Create a Custom Dataset Class: Define a custom dataset class that extends torch.utils.data.Dataset. This class should load and preprocess the images from your dataset.

3. Use Data Transforms: Use pytorch transforms module to apply transformations (e.g., resizing, normalization) to your images. This can be done in your custom dataset class.

4. Create a Data Loader: Use pytorch's DataLoader class to create a data loader that can iterate over your dataset in batches.

5 Define Your Model: Define your machine learning model using pytorch torch.nn module.
- A Convolutional Neural Network (CNN) is commonly used for image classification tasks.

6. Train Your Model: Use the data loader to train your model on the ASL dataset.

7. Evaluate Your Model: Evaluate the trained model on a separate test dataset to assess its performance. Use metrics such as accuracy, precision, recall, and F1 score.
"""
import torch
from torchvision import datasets, transforms
import torch.nn as nn

transform = transforms.Compose([
    transforms.Resize((244, 244)), # a common resizing size for images in machine learning tasks, particularly for convolutional neural networks (CNNs), is 224x224 pixels. 
    transforms.ToTensor() # convert images to pytorch tensors | neural networks operate on tensors, so this transformation is necessary to convert the image data into a format that the network can process.
])

# create dataset for training and testing
"datasets.ImageFolder is used to create datasets from folders containing images. It automatically assigns labels based on the folder structure, each subfolder in the specified folder is treated as a different class, and the images within each subfolder are assigned that class label"
"""
Why do we need a test/ train dataset variable?
separating the dataset into training and testing sets is a fundamental practice in machine learning for evaluating the model's performance. Heres why:

Training Dataset (train_dataset): is used to train the model by showing it examples along with their correct labels. The model learns to recognize patterns in the training data.

Testing Dataset (test_dataset): Used to evaluate the trained model's performance on unseen data. By testing on data that the model has not seen during training, we can assess how well the model generalizes to new, unseen examples.
"""
train_dataset = datasets.ImageFolder(root='data/archive/Train_Alphabet', transform=transform)
test_dataset = datasets.ImageFolder(root='data/archive/Test_Alphabet', transform=transform)

# create data loaders for training and testing
"""
NOTES: a data loader in pytorch is a utility that helps you load and iterate over your dataset in batches during the training or evaluation of a ml model. its part of pytorch's torch.utils.data module which provides tools for working with data in a way that is compatible with pytorchs's training loop.

breakdown of the parameters used when creating a data loader:

batch_size: This parameter specifies the number of samples that will be loaded and processed together in each iteration. Using batches allows you to take advantage of parallel processing capabilities of modern hardware, which can significantly speed up training.

NOTE: A typical batch size used in deep learning models, including Convolutional Neural Networks is often in the range of 32 to 128. However, the optimal batch size can vary depending on the specific dataset, model architecture, and available hardware. Smaller batch sizes are often used for models with limited memory or when training on larger datasets, while larger batch sizes can lead to faster training but may require more memory.

shuffle: When shuffle is set to True, the data loader shuffles the dataset at the beginning of each epoch (a complete pass through the dataset). Shuffling the data helps in preventing the model from learning the order of the examples in the dataset, which can lead to better generalization.

NOTE:For the train_loader, shuffle=True is used to shuffle the training dataset at the beginning of each epoch. This is important to ensure that the model doesn't learn spurious correlations based on the order of the data.

For the test_loader, shuffle=False is used because shuffling is not necessary during testing. We want the test dataset to remain in the same order to evaluate the model consistently on the same set of examples.
"""

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)