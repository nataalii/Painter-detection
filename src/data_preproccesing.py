from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch


def load_data(datapath):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor()
    ])
    
    dataset = ImageFolder(datapath, transform=transform)
    return dataset


def split_data(dataset, train_size=0.8):
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


