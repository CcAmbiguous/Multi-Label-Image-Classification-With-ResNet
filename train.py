import torch
import torch.optim as optim
from model import ResNetMultiLabelClassifier
from divdataloader import train_loader, val_loader
from loss import MultiLabelBCELoss
from loss import FocalLoss
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, optimizer, and loss function
model = ResNetMultiLabelClassifier(num_classes=80).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = MultiLabelBCELoss()
#criterion = FocalLoss()

# Training loop
num_epochs = 200
losses = []  # List to store the loss values

for epoch in range(num_epochs):
    epoch_loss = 0.0  # Initialize the loss for the epoch
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    average_epoch_loss = epoch_loss / len(train_loader)
    losses.append(average_epoch_loss)  # Save the average loss for the epoch

    print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {average_epoch_loss}')
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_labels).item()

    average_val_loss = val_loss / len(val_loader)
    print(f'Validation - Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_val_loss}')

# Save the trained model
torch.save(model.state_dict(), 'resnet_ML_model.pth')

# Plot the loss curve
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
