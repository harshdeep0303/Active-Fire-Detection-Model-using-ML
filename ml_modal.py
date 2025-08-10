import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

# Define the CNN model
class FireDetectionCNN(nn.Module):
    def __init__(self):
        super(FireDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Initializing with an incorrect value to show we will update it
        self.fc1 = nn.Linear(128 * 29 * 29, 512)  
        self.fc2 = nn.Linear(512, 2)  # Assuming binary classification (fire vs no fire)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        print("Shape after conv1 and pool:", x.shape)  # Debugging shape
        
        x = self.pool(F.relu(self.conv2(x)))
        print("Shape after conv2 and pool:", x.shape)  # Debugging shape
        
        x = self.pool(F.relu(self.conv3(x)))
        print("Shape after conv3 and pool:", x.shape)  # Debugging shape
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        print("Shape after flattening:", x.shape)  # Debugging shape
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, criterion, and optimizer
model = FireDetectionCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define transformations for the training set
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the images
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
])

# Load the dataset
train_dataset = datasets.ImageFolder(root='path_to_train_dataset', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.ImageFolder(root='path_to_test_dataset', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training function
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, 11):  # Training for 10 epochs
    train(model, device, train_loader, optimizer, criterion, epoch)

# Evaluation function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')

test(model, device, test_loader)

# Save the model
torch.save(model.state_dict(), 'fire_detection_model.pt')
