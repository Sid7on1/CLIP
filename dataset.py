from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageTextDataset(Dataset):
    def __init__(self, data, tokenizer, image_transform=None):
        self.data = data  # List of (image_path, caption)
        self.tokenizer = tokenizer
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, caption = self.data[idx]
        image = self.image_transform(Image.open(image_path).convert("RGB"))
        encoding = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }
