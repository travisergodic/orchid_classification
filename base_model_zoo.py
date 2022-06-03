import timm
import torch.nn as nn
import torchvision
from models import coatnet


def get_model(name, embed_size, drop_p=0.2):
    if name in ['resnet50', 'convnext_base']:
        base_model = getattr(torchvision.models, name)(pretrained=True)
     
    elif name[:8] == 'coatnet_':
        assert hasattr(coatnet, name), f"No model name: {name}"
        base_model = getattr(coatnet, name)(img_size)
        
    elif name == "facebook/convnext-base-384-22k-1k": 
        base_model = lambda x: transformers.ConvNextForImageClassification.from_pretrained(name)(x).logits

    else: 
        base_model = timm.create_model(name, pretrained=True)
        
    ####################################################################

    # remove fc
    if name == 'convnext_base':
        base_model.classifier = nn.Sequential(
            *(list(base_model.classifier)[:-1])
        )
    else:
        remove_fc(base_model)
        
    return base_model
