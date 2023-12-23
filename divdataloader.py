import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = self.data.iloc[idx, 1:].values
        labels = pd.to_numeric(labels, errors='coerce')

        # Handle non-numeric labels
        if pd.api.types.is_numeric_dtype(labels):
            labels = labels.astype('float32')
        else:
            labels = torch.zeros(len(labels), dtype=torch.float32)

        return image, torch.tensor(labels, dtype=torch.float32)



# Example of data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# train_dataset = CustomDataset(csv_file='data/cocodatas/train.csv', root_dir='data/cocodatas/train', transform=transform)
train_dataset = CustomDataset(csv_file='data/cocodatas/train.csv', root_dir='data/cocodatas/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = CustomDataset(csv_file='data/cocodatas/test.csv', root_dir='data/cocodatas/test', transform=transform)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = CustomDataset(csv_file='data/cocodatas/val.csv', root_dir='data/cocodatas/val', transform=transform)
val_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)