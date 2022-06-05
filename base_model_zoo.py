import timm
import torch.nn as nn
import torchvision

def build_resnetv2_50d_gn(raw_model): 
    setattr(
        raw_model.head,
        'fc',
        nn.Identity()
    )
    return raw_model, 2048

def build_resnetv2_50x1_bitm_in21k(raw_model):
    setattr(
        raw_model.head,
        'fc',
        nn.Identity()
    )
    return raw_model, 2048

def build_resnetv2_50x3_bitm(raw_model): 
    setattr(
        raw_model.head,
        'fc',
        nn.Identity()
    )
    return raw_model, 6144


def build_resnetv2_101x1_bitm_in21k(raw_model): 
    setattr(
        raw_model.head,
        'fc',
        nn.Identity()
    )
    return raw_model, 2048

