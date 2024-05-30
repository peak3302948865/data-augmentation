import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    writer.add_scalar('training loss', running_loss / len(train_loader), epoch)

def validate(model, val_loader, criterion, epoch, writer):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total
    writer.add_scalar('validation loss', val_loss / len(val_loader), epoch)
    writer.add_scalar('validation accuracy', val_accuracy, epoch)

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    # 将参数分为两组
    optimizer = optim.SGD([
        {'params': model.fc.parameters(), 'lr': 1e-2},
        {'params': [param for name, param in model.named_parameters() if 'fc' not in name], 'lr': lr}
    ], momentum=0.9)

    writer = SummaryWriter('runs/exp1')

    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer)
        validate(model, val_loader, criterion, epoch, writer)

    torch.save(model.state_dict(), 'model_weights.pth')
    writer.close()