from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional layers with batch normalization and max pooling
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # 1x100x100 -> 8x100x100
        self.bn1 = nn.BatchNorm2d(8)  # Batch normalization
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduces the size by half -> 8x50x50
        self.drop1 = nn.Dropout(0.1)  # Dropout to prevent overfitting

        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # 8x50x50 -> 16x50x50
        self.bn2 = nn.BatchNorm2d(16)  # Batch normalization
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x50x50 -> 16x25x25
        self.drop2 = nn.Dropout(0.1)  # Dropout

        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)  # 16x25x25 -> 32x25x25
        self.bn3 = nn.BatchNorm2d(32)  # Batch normalization
        self.pool3 = nn.MaxPool2d(2, 2)  # 32x25x25 -> 32x12x12
        self.drop3 = nn.Dropout(0.1)  # Dropout

        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)  # 32x12x12 -> 32x12x12
        self.bn4 = nn.BatchNorm2d(32)  # Batch normalization
        self.pool4 = nn.MaxPool2d(2, 2)  # 32x12x12 -> 32x6x6
        self.drop4 = nn.Dropout(0.05)  # Dropout

        # Global Average Pooling to reduce dimensions to 1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 32x6x6 -> 32x1x1

        # Fully connected layer
        self.fc = nn.Linear(32, 10)  # Output to 10 classes

    def forward(self, x):
        # Convolution with batch normalization, max pooling, ReLU activation, and dropout
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> MaxPool -> BatchNorm
        x = self.drop1(x)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> MaxPool -> BatchNorm
        x = self.drop2(x)

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> MaxPool -> BatchNorm
        x = self.drop3(x)

        x = self.pool4(F.relu(self.bn4(self.conv4(x))))  # Conv4 -> MaxPool -> BatchNorm
        x = self.drop4(x)

        # Global Average Pooling
        x = self.gap(x)

        # Flatten and pass through the fully connected layer
        x = torch.flatten(x, 1)  # Flatten the output from the GAP layer
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


# Model summary
model = Net()
print(model)

from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 32, 32))

torch.manual_seed(1)
batch_size = 32

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 20):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
