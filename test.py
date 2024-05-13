import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

from network import CNN

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224)), # resize doc: If size is an int, smaller edge of the image will be matched to this number.
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets using ImageFolder
test_dataset = torchvision.datasets.ImageFolder(root='datasets/test', transform=transform)

# Define data loaders
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print the number of samples in each dataset and each class
print('Dataset\t', 'Total\t', test_dataset.classes[0], '', test_dataset.classes[1], '', test_dataset.classes[2])
print('Test:\t', len(test_dataset), '\t', test_dataset.targets.count(0), '\t', test_dataset.targets.count(1), '\t', test_dataset.targets.count(2))
print()

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, '\n')

# Create an instance of the model
num_classes = len(test_dataset.classes)
model = CNN(num_classes=num_classes).to(device)

model.load_state_dict(torch.load('model.pth'))
model.eval()

# Test the model
correct = 0
total = 0

progress_bar = tqdm(total=len(test_loader), desc='Testing', position=0, leave=True)

with torch.no_grad():
    for inputs, labels in test_loader:
        logits = model.forward(inputs)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.update(1)

progress_bar.close()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
