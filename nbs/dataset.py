import os 
from torch.utils.data import Dataset
from PIL import Image

class LEVIRCDDataset(Dataset):
    def __init__(self, image_dir_A, image_dir_B, label_dir, transform=None):
        self.image_dir_A = image_dir_A
        self.image_dir_B = image_dir_B
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir_A)
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name_A = os.path.join(self.image_dir_A, self.image_filenames[idx])
        img_name_B = os.path.join(self.image_dir_B, self.image_filenames[idx])
        label_name = os.path.join(self.label_dir, self.image_filenames[idx])
        
        image_A = Image.open(img_name_A).convert("RGB")
        image_B = Image.open(img_name_B).convert("RGB")
        label = Image.open(label_name).convert("L") 
        
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
            label = self.transform(label)
        
        return (image_A, image_B), label