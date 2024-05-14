import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import sys

from network import CNN

model_name = 'model.pth'
if len(sys.argv) > 1:
    model_name = sys.argv[1]

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224)), # resize doc: If size is an int, smaller edge of the image will be matched to this number.
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets using ImageFolder
test_dataset = ImageFolder(root='datasets/test', transform=transform)

# Define the number of samples you want to use
subset_size = 624

# Create a random subset of indices
subset_indices = torch.randperm(len(test_dataset))[:subset_size]

# Create a DataLoader using SubsetRandomSampler
test_loader = DataLoader(test_dataset, batch_size=64, sampler=SubsetRandomSampler(subset_indices))

# Print the number of samples in each dataset and each class
print('Dataset\t', 'Test\t', test_dataset.classes[0], '', test_dataset.classes[1], '', test_dataset.classes[2])
print('Total:\t', len(test_dataset), '\t', test_dataset.targets.count(0), '\t', test_dataset.targets.count(1), '\t', test_dataset.targets.count(2), '\n')

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, '\n')

# Create an instance of the model
num_classes = len(test_dataset.classes)
model = CNN(num_classes=num_classes).to(device)

print('Loading the model', model_name + '...')
model.load_state_dict(torch.load(model_name))
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
