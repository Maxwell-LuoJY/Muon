import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
from muon import SingleDeviceMuon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


optimizers = {
    'Adam': optim.Adam
    #'SGD': optim.SGD,
    #'RMSprop': optim.RMSprop
}

results = {}

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = SingleDeviceMuon(list(model.parameters()))

print('Start training using Muon optimizer.')
start_time = time.time()

model.train()
for epoch in range(5):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

end_time = time.time()

training_time = end_time - start_time

model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

accuracy = correct / len(test_dataset)
results['Muon'] = {'accuracy': accuracy, 'time': training_time}

for name, optimizer in optimizers.items():
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    if name == 'Muon':
        # 将模型参数转换为列表形式
        optimizer_instance = optimizer(list(model.parameters()), lr=0.02, weight_decay=0, momentum=0.95)
    else:
        optimizer_instance = optimizer(model.parameters())

    print(f'Start training using {name} optimizer.')
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    model.train()
    for epoch in range(5):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_instance.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_instance.step()
    
    # 记录结束时间
    end_time = time.time()
    
    # 计算训练用时
    training_time = end_time - start_time

    # 评估模型
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(test_dataset)
    results[name] = {'accuracy': accuracy, 'time': training_time}

print("Optimizer Performance:")
for optimizer, metrics in results.items():
    print(f"{optimizer}: Accuracy = {metrics['accuracy']:.4f}, Training Time = {metrics['time']:.2f} seconds")