import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir,transform=None):
        if not os.path.exists(image_dir):
            raise FileNotFoundError (f"[!] Image directory does not exist: '{image_dir}'.\nPlease download the KITTI Road dataset and place it in this path.")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"[!] Mask directory not found: '{mask_dir}'.\nPlease make sure ground truth masks are placed correctly.")
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        mask_name = self.generate_mask_name(image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')


        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0.3).float()
        return image, mask

    def generate_mask_name(self,image_name):
        prefix,suffix = image_name.split('_')
        suffix = suffix.split(".")[0]
        return f"{prefix}_road_{suffix}.png"

