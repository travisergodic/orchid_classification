## configuration
import os
import torch 
import torchvision
from torchvision import transforms
from torch import nn

# basic 
img_size = (448, 448)
class_num = 219
device = "cuda" if torch.cuda.is_available() else "cpu"

# data augmentation
image_transform = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)

# model: resnet50 example 
is_arcface = True
is_mixup = False
drop_p = 0.2
embed_size = 256
model_name = 'resnet50'

# train
lr = 1e-4
batch_size = 32
num_epoch = 20
loss_fn = nn.NLLLoss()
decay_fn = lambda n: 1 if n <=15 else 0.2 
save_path = os.path.join("./", "classification_resnet50.pth")
best_path = os.path.join("./", "classification_resnet50_best.pth")

def Accuracy(predictions, targets):
    return (predictions.argmax(dim=1) == targets).sum()/targets.shape[0]

metric_dict = {
    "Accuracy": Accuracy,
    "CE": nn.NLLLoss()
}

save_config = {
    "path": save_path,
    "freq": 1,
    "best_path": best_path,
    "metric": "Accuracy"
}
