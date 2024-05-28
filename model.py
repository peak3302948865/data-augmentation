import torchvision.models as models
import torch.nn as nn

def get_model(num_classes=200, pretrained=True):
    # 加载预训练的ResNet-18模型
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
