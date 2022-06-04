import math
import torch
import torch.nn as nn

__all__ = ['CUSTOM_LAYER', 'get_base_model', 'Model']

# base_model
def get_base_model(model_dict):
    raw_model = model_dict['model_cls'](**{k:model_dict[k] for k in model_dict if k != 'model_cls'})
    for attr in ['fc', 'head', 'classifier']: 
        if hasattr(raw_model, attr):
            if isinstance(getattr(raw_model, attr), nn.Sequential):
                in_features = getattr(raw_model, attr).fc.in_features
                setattr(
                    raw_model,
                    attr,
                    nn.Sequential(*list(getattr(raw_model, attr))[:-2])
                )
                
            else: 
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
    def __init__(self, embed_size, num_classes):
        super().__init__()
        self.ffn = nn.Linear(embed_size, num_classes)
    
    def forward(self, embedding, targets=None): 
        return self.ffn(embedding)

    
# Arcface: arcface 不支援與 mixup, cutmix 混用 
class Arcface(nn.Module):
    def __init__(self, embed_size, num_classes, s=2, m=0.05, eps=1e-12):
        super().__init__()
        self._n_classes = num_classes
        self._embed_dim = embed_size
        self._s = float(s)
        self._m = float(m)
        self.eps = eps
        self.kernel = nn.Parameter(torch.Tensor(embed_size, num_classes))
        torch.nn.init.xavier_uniform_(self.kernel)
        
    def forward(self, embedding, targets):  # target: 
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
    

# # arcface with mixup
# class Arcface_Mixup(nn.Module):
#     def __init__(self, num_classes, embed_size, s=2, m=0.05, eps=1e-12):
#         super().__init__()
#         self._n_classes = num_classes
#         self._embed_dim = embed_size
#         self._s = float(s)
#         self._m = float(m)
#         self.eps = eps
#         # weights
#         self.kernel = nn.Parameter(torch.Tensor(embed_size, num_classes))
#         torch.nn.init.xavier_uniform_(self.kernel)
        
#     def forward(self, embedding, **kwargs): # target_a=None, target_b=None, lam=None): 
#         embedding = nn.functional.normalize(embedding, dim=1)
#         kernel_norm = nn.functional.normalize(self.kernel, dim=0)
#         cos_theta = torch.mm(embedding, kernel_norm)
        
#         if not self.training:
#             return self._s * cos_theta
#         else: 
#             target_a, target_b, lam = map(lambda x: kwargs[x], ('target_a', 'target_b', 'lam'))
#             one_hot_a = nn.functional.one_hot(target_a, num_classes=self._n_classes)
#             one_hot_b = nn.functional.one_hot(target_b, num_classes=self._n_classes) 
#             one_hot_labels =  one_hot_a + one_hot_b
#             margin = self._m * (lam * one_hot_a + (1-lam) * one_hot_b)
#             del one_hot_a, one_hot_b 
#             theta = torch.acos(torch.clip(cos_theta, -1+self.eps, 1-self.eps))
#             selected_labels = torch.where(torch.gt(theta, math.pi - self._m),
#                                           torch.zeros_like(one_hot_labels),
#                                           one_hot_labels)
#             final_theta = torch.where(selected_labels > 0, theta + margin, theta)
#             return self._s * torch.cos(final_theta)

CUSTOM_LAYER = {
    'arcface': Arcface, 
    'ffn': FFN
}