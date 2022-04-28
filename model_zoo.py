import timm
import torch.nn as nn
import torchvision

def get_model(name, embed_size, drop_p=0.2): 
    base_model = timm.create_model(name)

    if name == 'vit_small_patch16_384':        
        base_model.head = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(384, embed_size, bias=True)
            )

    if name == "swin_base_patch4_window12_384_in22k":
        base_model.head = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(1024, embed_size, bias=True)
        )

    if name == "efficientnetv2_rw_s": 
        base_model.classifier = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(1792, embed_size, bias=True)
        )

    if name == 'resnet50':
        base_model.fc = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(2048, embed_size, bias=True)
        )
        
    return base_model
