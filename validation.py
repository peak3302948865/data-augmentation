import torch
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

def main(data_dir, weight_path):
    _, val_loader = get_data_loaders(data_dir, batch_size=32)
    model = get_model()
    model = load_model_weights(model, weight_path)
    accuracy = calculate_accuracy(model, val_loader)
    print(f'Validation Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    data_dir = '/home/wukai/image-division/CUB_200_2011'
    weight_path = 'model_weights.pth'
    main(data_dir, weight_path)
