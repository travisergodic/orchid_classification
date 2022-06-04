import os
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import FlowerDataset, build_train_transform, build_test_transform
from trainer import Trainer
from hooks import *
from model_factory import *
from metric import * 
from loss import * 

parser = argparse.ArgumentParser(description="Image classification for flowers!")
parser.add_argument("--config_file", type=str)
args = parser.parse_args()

assert os.path.isfile(os.path.join("./configs", args.config_file))

# create configs/config.py
with open(os.path.join("./configs", args.config_file), 'r') as f: 
    text = f.read()
with open('./configs/config.py', 'w') as f:
    f.write(text)

from configs.config import *

## dataset & dataloader
df_train_label = pd.read_csv('./Labels/train_label.csv')
df_test_label = pd.read_csv('./Labels/test_label.csv')

# dataset
train_dataset = FlowerDataset('./training', df_train_label, class_num, build_train_transform(img_size))
test_dataset = FlowerDataset('./training', df_test_label, class_num, build_test_transform(img_size))

# dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# create model
if checkpoint_path is not None: 
    model = torch.load(checkpoint_path).to(DEVICE)
    print(f'Load model from {checkpoint_path} successfully!')
else:
    ## backbone 
    base_model, in_features = get_base_model(base_model_dict)
    ## custim layer 
    custom_layer_cls = CUSTOM_LAYER[custom_layer_config_dict.pop('layer_cls')]
    custom_layer_config_dict = {
        **custom_layer_config_dict, 
        **{'num_classes': class_num, 'embed_size': in_features}
    }
    custom_layer = custom_layer_cls(**custom_layer_config_dict)
    ## final model 
    model = Model(base_model, custom_layer, hugging_face=hugging_face).to(DEVICE)

# distributed 
if distributed: 
    model = nn.DataParallel(model)

# train_model
start = time.time()

## get iter_hook_cls
iter_hook_cls = ITER_HOOK.get(regularization_option, ITER_HOOK['normal'])
iter_hook_obj = iter_hook_cls()
print(f"Use iter hook of type <class {iter_hook_cls.__name__}> during training!")

## get loss
loss_cls = LOSS.get(loss_config.pop('loss_cls'))
loss_obj = loss_cls(**loss_config)

## get metric 
metric_dict = {metric: METRIC[metric] for metric in metric_list}
train_pipeline = Trainer(optim_dict, decay_fn, loss_obj, metric_dict, iter_hook_obj(), DEVICE)
train_pipeline.fit(model, train_dataloader, test_dataloader, num_epoch, save_config)
print(f"Training takes {time.time() - start} seconds!")
