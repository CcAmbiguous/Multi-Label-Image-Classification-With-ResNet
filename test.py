import torch
from model import ResNetMultiLabelClassifier
from divdataloader import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Load the trained model
model = ResNetMultiLabelClassifier(num_classes=80)
model.load_state_dict(torch.load('resnet_ML_model.pth'))

# Specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  
model.eval()

# Example of data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create a test dataset and data loader
test_dataset = CustomDataset(csv_file='data/cocodatas/test.csv', root_dir='data/cocodatas/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Testing loop
correct_predictions = 0
total_samples = 0
current_idx = 0  # 用于记录当前批次在数据集中的起始索引

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)  # 将标签也移到同一设备
        outputs = model(inputs)

        # Convert outputs to binary predictions (0 or 1)
        predictions = (outputs > 0.5).float()

        # Print the results
        for i in range(len(predictions)):
            idx = current_idx + i
            if idx < len(test_dataset.data):  # 确保不越界
                print(f"Image: {test_dataset.data.iloc[idx, 0]}, Predictions: {predictions[i].cpu().numpy()}, Ground Truth: {labels[i].cpu().numpy()}")

        # 更新索引
        current_idx += len(predictions)

        # Handle mismatched dimensions
        if predictions.shape[1] != labels.shape[1]:
            predictions = predictions[:, :labels.shape[1]]

        # 计算正确预测的数量（逐元素比较）
        correct_predictions += torch.sum(predictions == labels).item()
        total_samples += labels.numel()

# Calculate accuracy（注意：多标签分类中这个指标可能不理想）
accuracy = correct_predictions / total_samples if total_samples > 0 else 0

print(f"Accuracy: {accuracy * 100:.2f}%")
