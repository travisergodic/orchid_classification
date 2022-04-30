## configuration
import os
import torch 
import torchvision
from torchvision import transforms
from torch import nn

# basic 
img_size = (384, 384)
class_num = 219
device = "cuda" if torch.cuda.is_available() else "cpu"

# data augmentation
test_image_transform = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)

train_image_transform = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.RandomAffine(degrees=(-15., 1.), translate=(0.1, 0.1)),
        # transforms.RandomHorizontalFlip(p=0.2),
        # transforms.RandomVerticalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)


# model: resnet50 example 
is_sam = False
is_arcface = False
is_mixup = False
drop_p = 0.2
embed_size = 1024
model_name = 'facebook/convnext-base-384-22k-1k'

# train
lr = 1e-4
batch_size = 16
num_epoch = 100
loss_fn = nn.NLLLoss()
weight_decay = 0
decay_fn = lambda n: 1 # if n <=20 else 0.2 
save_path = os.path.join("./", "classification_efficient.pth")
best_path = os.path.join("./", "classification_efficient_best.pth")

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
