import math
import torch
import torch.nn as nn


# final model 
class Model(nn.Module):
    def __init__(self, base_model, custom_layer, logits=True): 
        super().__init__()
        self.base_model = base_model
        self.logsoftmax = nn.LogSoftmax(dim=1) 
        self.custom_layer = custom_layer
        self.logits = logits
        
    def forward(self, data, **kwargs): # target_a=None, target_b=None, lam=None):   
        embedding = self.base_model(data).logits if self.logits else self.base_model(data)
        embedding = self.custom_layer(embedding, **kwargs) # target_a, target_b, lam)
        return self.logsoftmax(embedding)


# FFN
class FFN(nn.Module): 
    def __init__(self, embed_size, num_classes):
        super().__init__()
        self.ffn = nn.Linear(embed_size, num_classes)
    
    def forward(self, embedding): 
        return self.ffn(embedding)

    
# ordinary arcface  
class Arcface(nn.Module):
    def __init__(self, num_classes, embed_size, s=2, m=0.05, eps=1e-12):
        super().__init__()
        self._n_classes = num_classes
        self._embed_dim = embed_size
        self._s = float(s)
        self._m = float(m)
        self.eps = eps
        # weights
        self.kernel = nn.Parameter(torch.Tensor(embed_size, num_classes))
        torch.nn.init.xavier_uniform_(self.kernel)
        
    def forward(self, embedding, **kwargs):  # target): 
        embedding = nn.functional.normalize(embedding, dim=1)
        kernel_norm = nn.functional.normalize(self.kernel, dim=0)
        cos_theta = torch.mm(embedding, kernel_norm)
        
        if not self.training:
            return self._s * cos_theta
        else: 
            target = kwargs['target']
            one_hot_labels = nn.functional.one_hot(target, num_classes=self._n_classes)
            theta = torch.acos(torch.clip(cos_theta, -1+self.eps, 1-self.eps))
            selected_labels = torch.where(torch.gt(theta, math.pi - self._m),
                                          torch.zeros_like(one_hot_labels),
                                          one_hot_labels)
            final_theta = torch.where(selected_labels > 0, 
                                      theta + self._m,
                                      theta)
            return self._s * torch.cos(final_theta)    
    

# arcface with mixup
class Arcface_Mixup(nn.Module):
    def __init__(self, num_classes, embed_size, s=2, m=0.05, eps=1e-12):
        super().__init__()
        self._n_classes = num_classes
        self._embed_dim = embed_size
        self._s = float(s)
        self._m = float(m)
        self.eps = eps
        # weights
        self.kernel = nn.Parameter(torch.Tensor(embed_size, num_classes))
        torch.nn.init.xavier_uniform_(self.kernel)
        
    def forward(self, embedding, **kwargs): # target_a=None, target_b=None, lam=None): 
        embedding = nn.functional.normalize(embedding, dim=1)
        kernel_norm = nn.functional.normalize(self.kernel, dim=0)
        cos_theta = torch.mm(embedding, kernel_norm)
        
        if not self.training:
            return self._s * cos_theta
        else: 
            target_a, target_b, lam = map(lambda x: kwargs[x], ('target_a', 'target_b', 'lam'))
            one_hot_a = nn.functional.one_hot(target_a, num_classes=self._n_classes)
            one_hot_b = nn.functional.one_hot(target_b, num_classes=self._n_classes) 
            one_hot_labels =  one_hot_a + one_hot_b
            margin = self._m * (lam * one_hot_a + (1-lam) * one_hot_b)
            del one_hot_a, one_hot_b 
            theta = torch.acos(torch.clip(cos_theta, -1+self.eps, 1-self.eps))
            selected_labels = torch.where(torch.gt(theta, math.pi - self._m),
                                          torch.zeros_like(one_hot_labels),
                                          one_hot_labels)
            final_theta = torch.where(selected_labels > 0, theta + margin, theta)
            return self._s * torch.cos(final_theta)
        

if __name__ == '__main__': 
    # hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_arcface = True
    is_mixup = True
    num_classes = 219
    embed_size = 256
    
    # base model: resnet50 example 
    base_model = torchvision.models.resnet50(pretrained=True)
    base_model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(2048, 256, bias=True)
    )
 
    # custom layer 
    if is_arcface: 
        custom_layer = Arcface_Mixup(num_classes, embed_size) if is_mixup else Arcface(num_classes, embed_size)
    else: 
        custom_layer = FFN(embed_size, num_classes)
        
    # final model 
    model = Model(base_model, custom_layer).to(device)
