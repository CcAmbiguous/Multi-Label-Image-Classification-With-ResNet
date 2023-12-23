import torch
from model import ResNetMultiLabelClassifier
from divdataloader import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Load the trained model
model = ResNetMultiLabelClassifier(num_classes=80)
model.load_state_dict(torch.load('resnet_ML_model.pth'))
model.eval()

# Specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)

        # Convert outputs to binary predictions (0 or 1)
        predictions = (outputs > 0.5).float()

        # Print the results
        for i in range(len(predictions)):
            print(
                f"Image: {test_dataset.data.iloc[i, 0]}, Predictions: {predictions[i].cpu().numpy()}, Ground Truth: {labels[i].cpu().numpy()}")

        # Handle mismatched dimensions
        if predictions.shape[1] != labels.shape[1]:
            # Resize predictions to match the number of classes
            predictions = predictions[:, :labels.shape[1]]

        # Update counts
        correct_predictions += torch.sum(predictions == labels[:, :predictions.shape[1]]).item()
        total_samples += labels.numel()



# Calculate accuracy
accuracy = correct_predictions / total_samples

print(f"Accuracy: {accuracy * 100:.2f}%")

