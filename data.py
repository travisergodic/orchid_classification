import os
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

## dataset
class FlowerDataset(Dataset):
    def __init__(self, image_dir, df_label, num_classes, image_transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.image_paths = list(df_label['filename'])
        self.image_labels = list(df_label['category'])
        self.num_classes = num_classes
        self.image_transform = image_transform 
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        label = self.image_labels[index]
      
        image = Image.open(image_path).convert('RGB')
        
        if self.image_transform is not None:
            image = self.image_transform(image)

        label = nn.functional.one_hot(label, num_classes=self.num_classes)
        return image, label

    def __len__(self):
        return len(self.image_paths)


# transform 
build_test_transform = lambda img_size: transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)

build_train_transform = lambda img_size: transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.RandomAffine(degrees=(-15., 1.), translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)