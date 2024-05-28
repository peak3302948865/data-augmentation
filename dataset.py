import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class CUB200Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        
        # 读取文件
        with open(os.path.join(root, 'train_test_split.txt'), 'r') as f:
            train_test_split = f.readlines()
        
        with open(os.path.join(root, 'image_class_labels.txt'), 'r') as f:
            image_class_labels = f.readlines()
        
        with open(os.path.join(root, 'images.txt'), 'r') as f:
            images = f.readlines()

        # 解析文件
        self.samples = []
        image_paths = {int(line.split()[0]): line.split()[1] for line in images}
        for i, line in enumerate(train_test_split):
            image_id, is_train = line.strip().split()
            image_id = int(image_id)
            if (train and int(is_train)) or (not train and not int(is_train)):
                image_path = os.path.join(root, 'images', image_paths[image_id])
                label = int(image_class_labels[i].strip().split()[1]) - 1
                self.samples.append((image_path, label))
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __len__(self):
        return len(self.samples)

    @staticmethod
    def loader(path):
        return Image.open(path).convert('RGB')

def get_data_loaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    train_dataset = CUB200Dataset(root=data_dir, train=True, transform=transform)
    val_dataset = CUB200Dataset(root=data_dir, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == '__main__':
    data_dir = '/home/wukai/image-division/CUB_200_2011'
    train_loader, val_loader = get_data_loaders(data_dir)

    # 打印一些训练数据的示例
    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels: {labels}")
        break
