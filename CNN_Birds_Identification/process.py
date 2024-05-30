import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataset import get_data_loaders
from model import get_model

def load_model_weights(model, weight_path):
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model

def calculate_accuracy(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    writer.add_scalar('training loss', running_loss / len(train_loader), epoch)
    writer.add_scalar('training accuracy', accuracy, epoch)
    print(f'Epoch {epoch}, Training Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {accuracy:.2f}%')

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

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    writer.add_scalar('validation loss', val_loss / len(val_loader), epoch)
    writer.add_scalar('validation accuracy', accuracy, epoch)
    print(f'Epoch {epoch}, Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {accuracy:.2f}%')

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': model.fc.parameters(), 'lr': 1e-2},
        {'params': model.parameters(), 'lr': lr}
    ], momentum=0.9)

    writer = SummaryWriter('runs/exp1')

    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer)
        validate(model, val_loader, criterion, epoch, writer)

    torch.save(model.state_dict(), 'model_weights.pth')
    writer.close()

def main(data_dir, weight_path=None, num_epochs=10, lr=1e-4, batch_size=32):
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=batch_size)
    model = get_model()

    if weight_path:
        model = load_model_weights(model, weight_path)
        accuracy = calculate_accuracy(model, val_loader)
        print(f'Validation Accuracy: {accuracy:.2f}%')
    else:
        train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train or evaluate the model.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory for dataset')
    parser.add_argument('--weight_path', type=str, help='Path to the model weights')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')

    args = parser.parse_args()
    main(args.data_dir, args.weight_path, args.num_epochs, args.lr, args.batch_size)
