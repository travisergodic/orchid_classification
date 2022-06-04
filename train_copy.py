import os
import time
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from data import FlowerDataset, build_train_transform, build_test_transform
from trainer import Trainer
from hooks import Iter_Hook_dict, Normal_Iter_Hook
from model_factory import *
import torch.nn as nn
import torch

def train(): 
    ## dataset & dataloader
    df_train_label = pd.read_csv('./train_label.csv')
    df_test_label = pd.read_csv('./test_label.csv')

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
        base_model, in_features = get_base_model(base_model_dict)
        CUSTOM_LAYER_CLS = CUSTOM_LAYER_dict[custom_layer_config_dict.pop('layer_cls')]
        custom_layer_config_dict = {
            **custom_layer_config_dict, 
            **{'num_classes': class_num, 'embed_size': in_features}
        }
        custom_layer = CUSTOM_LAYER_CLS(**custom_layer_config_dict)
        model = Model(base_model, custom_layer, hugging_face=hugging_face).to(DEVICE)

    # distributed 
    if distributed: 
        model = nn.DataParallel(model)

    # train_model
    start = time.time()

    ## get iter_hook_cls
    Iter_Hook_CLS = Iter_Hook_dict.get(regularization_option, Normal_Iter_Hook)

    print(f"Use iter hook of type <class {Iter_Hook_CLS.__name__}> during training!")
    train_pipeline = Trainer(optim_dict, decay_fn, loss_fn, metric_dict, Iter_Hook_CLS(), DEVICE)
    train_pipeline.fit(model, train_dataloader, test_dataloader, num_epoch, save_config)
    print(f"Training takes {time.time() - start} seconds!")


if __name__ == '__main__': 
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
    train()
    os.remove('./configs/config.py')

