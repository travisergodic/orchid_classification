import ttach as tta
import torch
import torch.nn as nn 

test_img_size_list = [(384, 384)] 
tta_fn = tta.aliases.flip_transform()
activation = nn.Softmax(dim=1)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
metric = 'mix_score'