import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

from network import CNN

transform = transforms.Compose([
     transforms.Resize((224)), # resize doc: If size is an int, smaller edge of the image will be matched to this number.
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets using ImageFolder
train_dataset = torchvision.datasets.ImageFolder(root='datasets/train', transform=transform)

# Define data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Print the number of samples in each dataset and each class
print('Dataset\t', 'Total\t', train_dataset.classes[0], '', train_dataset.classes[1])
print('Train:\t', len(train_dataset), '\t',  train_dataset.targets.count(0), '\t',  train_dataset.targets.count(1), '\n')

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, '\n')

# Create an instance of the model
num_classes = len(train_dataset.classes)
model = CNN(num_classes=num_classes).to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
train_loss = []
accuracy_total_train = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(total=len(train_loader), desc='Epoch {}/{}'.format(epoch+1, num_epochs), position=0, leave=True)

    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
        running_loss += loss.item() * inputs.size(0)

        # Calculate the accuracy
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        accuracy_total_train.append(torch.sum(preds == labels.data).item() / float(inputs.size(0)))

        progress_bar.update(1)

    epoch_loss = running_loss / len(train_dataset)
    progress_bar.close()
    print('Loss: {:.4f}'.format(epoch_loss),
          'Accuracy: {:.4f}'.format(sum(accuracy_total_train) / len(accuracy_total_train)))


print('Finished Training')

print('Saving the model...')
torch.save(model.state_dict(), 'model.pth')
print('Model saved as model.pth')
