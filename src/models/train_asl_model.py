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
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        print(f"input shape: {inputs.shape}")
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save your trained model if needed
torch.save(model.state_dict(), 'asl_model.pth')
