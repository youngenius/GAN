import torch
import os
import torchvision
import torchvision.transforms as transforms

class Dataset:
    def __init__(self, flags):
        self.flags = flags
        self.data_path = os.path.join(self.flags.data, 'face')

    def load_dataset(self):
        train_dataset = torchvision.datasets.ImageFolder(
            root=self.data_path,
            transform = transforms.Compose([
                transforms.CenterCrop(400),
                transforms.Resize(64), #256
                transforms.ToTensor(),
            ])
            #transform=torchvision.transforms.ToTensor(),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.flags.batch,
            num_workers=0,
            shuffle=True
        )

        return train_loader
