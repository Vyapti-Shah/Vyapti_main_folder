import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DeepfakeDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Real and Fake images.
    Expects a directory structure like:
        dataset_dir/
            real/
                img1.jpg
                img2.png
            fake/
                fake1.jpg
                fake2.png
    """
    def __init__(self, root_dir: str, image_size: int = 256, is_train: bool = True):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.is_train = is_train
        
        self.image_paths = []
        self.labels = []
        
        # 0 = REAL, 1 = FAKE (AI GENERATED)
        self.class_map = {"real": 0, "fake": 1}
        
        # Load all paths
        for class_name, label in self.class_map.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"[!] Warning: Dataset directory {class_dir} does not exist.")
                continue
                
            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                    
        print(f"[*] Loaded {len(self.image_paths)} images from {root_dir}")
        
        # Basic Augmentations
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"[-] Error loading {img_path}: {e}")
            # Fallback to a zero tensor if image is corrupt
            image = torch.zeros((3, self.image_size, self.image_size))
            
        return image, torch.tensor(label, dtype=torch.long)
