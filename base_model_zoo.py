import timm
import torch.nn as nn
import torchvision
from models import coatnet
from config import img_size


def remove_fc(model): 
    attrs = ['fc', 'head', 'classifier']
    for attr in attrs: 
        if hasattr(model, attr): 
            getattr(model, attr) = nn.Identity()
            break

            
def get_model(name, embed_size, drop_p=0.2):
    if name in ['resnet50', 'convnext_base']:
        base_model = getattr(torchvision.models, name)(pretrained=True)
     
    elif name[:8] == 'coatnet_0':
        assert hasattr(coatnet, name), f"No model name: {name}"
        base_model = getattr(coatnet, name)(img_size)
        
    else: 
        base_model = timm.create_model(name, pretrained=True)
        
    ####################################################################
    
    if name == 'vit_small_patch16_384':        
        # base_model.head = nn.Sequential(
        #     nn.Dropout(p=drop_p),
        #     nn.Linear(384, embed_size, bias=True)
        #     )
        remove_fc = remove_fc(model)

    elif name == 'swin_base_patch4_window12_384_in22k':
        # base_model.head = nn.Sequential(
        #     nn.Dropout(p=drop_p),
        #     nn.Linear(1024, embed_size, bias=True)
        # )
        remove_fc = remove_fc(model)

    elif name == 'efficientnetv2_rw_s': 
        # base_model.classifier = nn.Sequential(
        #     nn.Dropout(p=drop_p),
        #     nn.Linear(1792, embed_size, bias=True)
        # )
        remove_fc = remove_fc(model)

    elif name == 'resnet50':
        # base_model.fc = nn.Sequential(
        #     nn.Dropout(p=drop_p),
        #     nn.Linear(2048, embed_size, bias=True)
        # )
        remove_fc = remove_fc(model)
        
    return base_model
