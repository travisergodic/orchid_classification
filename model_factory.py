import math
import torch
import torch.nn as nn
import base_model_zoo

__all__ = ['CUSTOM_LAYER', 'get_base_model', 'Model']


# base_model
def get_base_model(model_dict):
    raw_model = model_dict['model_cls'](**{k:model_dict[k] for k in model_dict if k != 'model_cls'})
    if hasattr(base_model_zoo , 'build_' + model_dict['model_name']): 
        return getattr(base_model_zoo, 'build_' + model_dict['model_name'])(raw_model)

    for attr in ['fc', 'head', 'classifier']: 
        if hasattr(raw_model, attr):
            if isinstance(getattr(raw_model, attr), nn.Sequential):
                print(getattr(raw_model, attr))
                in_features = getattr(raw_model, attr).fc.in_features
                setattr(
                    getattr(raw_model, attr),
                    'fc',
                    nn.Identity()
                )
                print(getattr(raw_model, attr))
            else: 
                print(getattr(raw_model, attr))
                in_features = getattr(raw_model, attr).in_features
                setattr(
                    raw_model, 
                    attr, 
                    nn.Identity()
                )
            return raw_model, in_features  


# final model 
class Model(nn.Module):
    def __init__(self, base_model, custom_layer, hugging_face=False): 
        super().__init__()
        self.base_model = base_model
        self.custom_layer = custom_layer
        self.hugging_face = hugging_face
        
    def forward(self, data, targets=None): # target_a=None, target_b=None, lam=None):   
        embedding = self.base_model(data)
        if self.hugging_face: 
            embedding = embedding.logits
        return self.custom_layer(embedding, targets) # target_a, target_b, lam)


# FFN
class FFN(nn.Module): 
    def __init__(self, embed_size, num_classes, hidden_dim=None, drop_p=0.):
        super().__init__()
        self.drop = nn.Dropout(p=drop_p)
        if hidden_dim is None:
            self.ffn = nn.Linear(embed_size, num_classes)

        else: 
            self.ffn = nn.Sequential(
                nn.Linear(embed_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
    
    def forward(self, embedding, targets=None):
        return self.ffn(self.drop(embedding))

    
# Arcface: arcface 不支援與 mixup, cutmix 混用 
class Arcface(nn.Module):
    def __init__(self, embed_size, num_classes, hidden_dim=None, s=2, m=0.05, eps=1e-12, drop_p=0.):
        super().__init__()
        self._n_classes = num_classes
        self._embed_dim = embed_size
        self._s = float(s)
        self._m = float(m)
        self.eps = eps
        self.drop = nn.Dropout(p=drop_p)
        if hidden_dim is None: 
            self.hidden_layer = nn.Identity()
            self.kernel = nn.Parameter(torch.Tensor(embed_size, num_classes))
        else: 
            self.hidden_layer = nn.Sequential(
                nn.Linear(embed_size, hidden_dim), 
                nn.ReLU()
            )
            self.kernel = nn.Parameter(torch.Tensor(hidden_dim, num_classes))
            
        torch.nn.init.xavier_uniform_(self.kernel)

    def forward(self, embedding, targets):
        embedding = self.hidden_layer(self.drop(embedding))
        embedding = nn.functional.normalize(embedding, dim=1)
        kernel_norm = nn.functional.normalize(self.kernel, dim=0)
        cos_theta = torch.mm(embedding, kernel_norm)
        
        if self.training:  
            theta = torch.acos(torch.clip(cos_theta, -1+self.eps, 1-self.eps))
            selected_labels = torch.where(torch.gt(theta, math.pi - self._m),
                                          torch.zeros_like(targets),
                                          targets)
            final_theta = torch.where(selected_labels > 0, 
                                      theta + self._m,
                                      theta)
            return self._s * torch.cos(final_theta)    
        return self._s * cos_theta

CUSTOM_LAYER = {
    'arcface': Arcface, 
    'ffn': FFN
}