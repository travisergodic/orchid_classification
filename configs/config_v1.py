import os
import torch 
from torch import optim
from loss import Loss
import transformers

# basic
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
distributed = False

# dataset & dataloader
img_size = (384, 384)
class_num = 219
num_workers = 2

# regularization 
regularization_option = 'sam'

# train
lr = 1e-4
batch_size = 16
num_epoch = 100
loss_fn = Loss()
weight_decay = 0
decay_fn = lambda n: 1 # if n <=20 else 0.2 
optim_dict = {
    'optim_cls': optim.AdamW, 
    'lr': 1e-4, 
    'weight_decay': 3e-3
}

#  model
checkpoint_path = None
model_dict = {
    'model_cls': transformers.ConvNextForImageClassification.from_pretrained,
    'name': 'facebook/convnext-base-384-22k-1k'
}
hugging_face = True
custom_layer_dict = {
    'name': 'arcface', 
    's': 2,
    'm': 0.05
}

# save path 
save_path = os.path.join("./checkpoints", "model_v1.pt")
best_path = os.path.join("./checkpoints", "model_v1_best.pt")

# metrics & loss
def Accuracy(predictions, targets):
    return (predictions.argmax(dim=1) == targets).sum()/targets.shape[0]

metric_dict = {
    "Accuracy": Accuracy,
    "CE": Loss()
}

save_config = {
    "path": save_path,
    "freq": 1,
    "best_path": best_path,
    "metric": "Accuracy"
}
