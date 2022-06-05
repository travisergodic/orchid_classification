import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F


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

        label = nn.functional.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return image, label

    def __len__(self):
        return len(self.image_paths)


# transform 
class Train_Preprocessor(nn.Module): 
    def __init__(self, img_size=None, h_flip_p=0.5, v_flip_p=0.5):
        super().__init__()
        if img_size is not None: 
            self.img_size = img_size
            self.resize_image = transforms.Resize(self.img_size, interpolation=InterpolationMode.BILINEAR)  
        else: 
            self.resize_image = nn.Identity()
            
        self.jitter = transforms.ColorJitter(0.3, 0.3, 0.3)
        self.blur = transforms.GaussianBlur((1, 3))
        self.perspect = transforms.RandomPerspective(distortion_scale=0.2, p=0.2)
        
        self.h_flip_p = h_flip_p
        self.v_flip_p = v_flip_p
        
        self.preprocess = transforms.Compose(
            [
              transforms.ToTensor(),
              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
    
    @torch.no_grad()
    def forward(self, img): 
        # random crop
        W, H = img.size
        w, h = random.randint(int(0.90*W), W), random.randint(int(0.90*H), H)
        i, j = random.randint(0, H-h), random.randint(0, W-w)
        img = F.crop(img, i, j, h, w)

        # resize & color transform 
        img = self.blur(self.jitter(self.resize_image(img)))

        # Random horizontal flipping
        if random.random() < self.h_flip_p:
            img = F.hflip(img)

        # Random vertical flipping
        if random.random() < self.v_flip_p:
            img = F.vflip(img)    

        # Random Perspective
        img = self.perspect(img)

        return self.preprocess(img)


Test_Preprocessor = lambda img_size: transforms.Compose(
    [
        transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)
