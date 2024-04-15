import torch
import torch.optim as optim
import torch.nn as nn
from src.datasets.asl_dataset import ASLDataset
from src.models.asl_model import ASLModel

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

test_dataset_path = "data/archive/Test_Alphabet"
train_dataset_path = "data/archive/Train_Alphabet"


train_dataset = ASLDataset(data_path=train_dataset_path, shuffle=True, batch_size=32)
test_dataset = ASLDataset(data_path=test_dataset_path, shuffle=False, batch_size=32)

train_loader = train_dataset.get_loader()
test_loader = test_dataset.get_loader()

model = ASLModel()

# Define your loss function and optimizer
"""
criterion = nn.CrossEntropyLoss(): This line creates an instance of the CrossEntropyLoss class from the nn module in PyTorch. CrossEntropyLoss is a common choice for classification tasks with multiple classes. It computes the softmax of the output and then calculates the cross-entropy loss between the predicted probabilities and the actual target labels.

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9): Here, you're creating an instance of the SGD (Stochastic Gradient Descent) optimizer. The model.parameters() method provides the parameters (weights and biases) of your model to the optimizer, so it knows which parameters to update during training. The lr=0.001 argument specifies the learning rate, which controls the step size of the optimizer. The momentum=0.9 argument adds momentum to the SGD update, which can help accelerate convergence and avoid local minima.
"""
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# model.train()

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     for inputs, labels in train_loader:
#         # print(f"input shape: {inputs.shape}")
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# # Save your trained model if needed
# torch.save(model.state_dict(), 'asl_model.pth')
# Load the trained model
model.load_state_dict(torch.load('asl_model.pth'))
model.eval()
classes = ('A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
correct = 0
total = 0

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the asl model to new pics: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
# # Iterate over a batch of images from the test loader
# for images, labels in test_loader:
#     # Predict labels for the batch of images
#     outputs = model(images)
#     _, predicted = torch.max(outputs, 1)
#     correct += (predicted == labels).sum().item()  # Count correct predictions
#     total += labels.size(0)  # Update total images based on batch size
#     # Compare predicted labels with actual labels
#     # for i in range(len(predicted)):
#     #     print(f'Predicted: {classes[predicted[i]]}, Actual: {classes[labels[i]]}')


# # After the loop
# accuracy = (correct / total) * 100
# print(f'Accuracy: {accuracy:.2f}%')
# dataiter = iter(test_loader)
# images, labels = next(dataiter)


# import matplotlib.pyplot as plt
# import numpy as np

# # functions to show an image

# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# dataiter = iter(test_loader)
# # get some random training images
# # images, labels = next(dataiter)

# # show images
# for images, labels in dataiter:
#   # Show the first num_images_to_show images

# # show images
#     imshow(torchvision.utils.make_grid(images))
#     print(' '.join(f'{classes[labels[j]]:5s}' for j in range(32)))
#     # imshow(torchvision.utils.make_grid(images[:num_images_to_show]))
#     # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))
#     # Only show images from one batch for now
# # Display a random number of images (up to 4 in this case)


# # imshow(torchvision.utils.make_grid(images))
# # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))