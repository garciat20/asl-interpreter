import torch
from torchvision import datasets, transforms

class ASLDataset(torch.utils.data.Dataset):
    __slots__ = ["dataset", "loader", "transform"]

    def __init__(self, data_path, shuffle, batch_size=32):
        self.transform = transforms.Compose([
            transforms.Resize((244, 244)), # a common resizing size for images in machine learning tasks, particularly for convolutional neural networks (CNNs), is 224x224 pixels. 
            transforms.ToTensor() # convert images to pytorch tensors | neural networks operate on tensors, so this transformation is necessary to convert the image data into a format that the network can process.
        ])
        self.dataset = datasets.ImageFolder(root=data_path, transform=self.transform)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def get_loader(self):
        return self.loader

# train_dataset = datasets.ImageFolder(root='data/archive/Train_Alphabet', transform=self.transform)
# test_dataset = datasets.ImageFolder(root='data/archive/Test_Alphabet', transform=self.transform)

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)