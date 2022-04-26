import timm
import torch.nn as nn
import torchvision

def get_model(name, embed_size, drop_p=0.2): 
    if name == 'vit_small_patch16_384':
        base_model = timm.create_model(name)
        base_model.head = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(384, embed_size, bias=True)
            )

    if name == 'resnet50':
        base_model = torchvision.models.resnet50(pretrained=True)
        base_model.fc = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(2048, embed_size, bias=True)
        )
    return base_model
