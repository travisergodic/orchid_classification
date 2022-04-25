import os
from PIL import Image
from torch.utils.data import Dataset


## transform 在 main 中定義
class FlowerDataset(Dataset):
    def __init__(self, image_dir, df_label, image_transform=None):
        self.image_dir = image_dir
        self.image_paths = list(df_label['filename'])
        self.image_labels = list(df_label['category'])
        self.image_transform = image_transform 
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        label = self.image_labels[index]
      
        image = Image.open(image_path).convert('RGB')
        
        if self.image_transform is not None:
            image = self.image_transform(image)
            
        return image, label

    def __len__(self):
        return len(self.image_paths)