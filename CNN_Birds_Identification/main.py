from dataset import get_data_loaders
from model import get_model
from train import train_model

data_dir = '/home/wukai/image-division/CUB_200_2011'

# 加载数据
train_loader, val_loader = get_data_loaders(data_dir)

# 获取模型
model = get_model()

# 训练模型
train_model(model, train_loader, val_loader)