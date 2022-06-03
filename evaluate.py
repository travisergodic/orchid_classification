import cv2
import os 
import torch 
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import ttach as tta

class Evaluator:
    def __init__(self, models, image_transforms, device='cuda', activation=nn.Softmax(dim=1)):
        models = models if isinstance(models, list) else [models]  
        assert all([isinstance(model, nn.Module) for model in models])
        self.models = [
            nn.Sequential(
                model,
                nn.Identity() if activation is None else activation 
            ).to(device) for model in models
        ]
    
        self.image_transforms = image_transforms
        self.device = device

    def _wrap_TTA(self, tta_transform): 
        if tta_transform:
            return [
                tta.SegmentationTTAWrapper(model, tta_transform, merge_mode='mean').eval() 
                for model in self.models
            ]
        return [model.eval() for model in self.models]
    
    @torch.no_grad()
    def _predict(self, models, path):
        ensemble_res = [] 
        img = Image.open(path).convert('RGB')
        for i, model in enumerate(models): 
            X = self.image_transforms[i](img).unsqueeze(0).to(self.device) 
            ensemble_res.append(model(X).squeeze())
        return sum(ensemble_res).argmax().item()
    
    def evaluate(self, image_paths, label_list, metric, tta_transform=False):
        if type(image_paths) == str: 
            assert os.path.isdir(image_paths), f"{image_paths} is not a directory!"
            image_paths = sorted([os.path.join(image_paths, basename) for basename in os.listdir(image_paths)])
        assert len(image_paths) == len(label_list)
        
        models = self._wrap_TTA(tta_transform)

        res_list = [] 
        for image_path in tqdm(image_paths): 
            res_list.append(self._predict(models, image_path))
        return getattr(Evaluator, metric)(res_list, label_list)          
    
    def make_prediction(self, paths, tta_transform=False): 
        if type(paths) == str: 
            assert os.path.isdir(paths), f"{paths} is not a directory!"
            img_dir = paths
            paths = [os.path.join(paths, basename) for basename in os.listdir(paths)]
            print(f"Find {len(paths)} files under {img_dir}")
        else: 
            assert all([os.path.isfile(path) for path in paths])

        models = self._wrap_TTA(tta_transform)
        
        res_list = [] 
        for path in tqdm(paths): 
            res_list.append(self._predict(models, path))
        
        pd.DataFrame(
            {
                'filename': [os.path.basename(path) for path in paths],
                'label': res_list
            }
        ).to('./predict_result.csv', index=False)

    @staticmethod
    def accuracy(preds, targets):
        assert len(preds) == len(targets)
        preds, targets = np.array(preds), np.array(targets)
        return sum(preds == targets)/len(preds)

    @staticmethod
    def f1_score(preds, targets): 
        pass
        