import ttach as tta
import torch
import torch.nn as nn 

test_img_size = (512, 512)
test_img_size_list = [(512, 512), (384, 384)] 
tta_fn = tta.aliases.five_crop_transform()
activation = nn.Softmax(dim=1)
image_dir = '/content/STAS-segmentation/Train_Images/'
label_dir = '/content/STAS-segmentation/Annotations/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
metric = 'accuracy'