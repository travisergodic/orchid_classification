import os
import torch 
from torch import optim
import transformers
import timm

# basic
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
distributed = False

# dataset & dataloader
img_size = (384, 384)
class_num = 219
num_workers = 2

# regularization 
regularization_option = 'normal'

# train
batch_size = 16
num_epoch = 100
decay_fn = lambda n: 1 # if n <=20 else 0.2 
optim_dict = {
    'optim_cls': optim.Adam, 
    'lr': 1e-4, 
    # 'weight_decay': 1e-3
}


# loss & metric
loss_config = {
    'loss_cls': 'Loss'
} 

metric_list = ['accuracy', 'mix_score']

#  model
checkpoint_path = None
base_model_dict = {
    'model_cls': timm.create_model,
    'model_name': 'convnext_large_384_in22ft1k', 
    'pretrained': True
}

hugging_face = False

custom_layer_config_dict = {
    'layer_cls': 'ffn'
}

# save path 
save_path = os.path.join("./checkpoints", "model_v1.pt")
best_path = os.path.join("./checkpoints", "model_v1_best.pt")
save_config = {
    "path": save_path,
    "freq": 1,
    "best_path": best_path,
    "metric": "mix_score"
}
