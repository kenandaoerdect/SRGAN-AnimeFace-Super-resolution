from torch.utils.data.dataset import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, train_img_path, transform=None):
        super(TrainDataset, self).__init__()
        self.img_hr = os.listdir(train_img_path + '/HR')
        self.img_lr = os.listdir(train_img_path + '/LR')
        self.transform = transform
        self.hr_and_lr = [] 
        assert len(self.img_hr) == len(self.img_lr), 'Number does not match'
        for i in range(len(self.img_hr)):
            self.hr_and_lr.append(
                    (os.path.join(train_img_path, 'HR', self.img_hr[i]),
                     os.path.join(train_img_path, 'LR', self.img_lr[i]))
                )

    def __getitem__(self, item):
        hr_path, lr_path = self.hr_and_lr[item]
        hr_arr = Image.open(hr_path)
        lr_arr = Image.open(lr_path)
        return np.array(lr_arr).transpose(2, 0, 1).astype(np.float32), np.array(hr_arr).transpose(2, 0, 1).astype(np.float32)

    def __len__(self):
        return len(self.img_hr)





if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data = TrainDataset('data/train', transform)
    print(len(data))
    data_loader = DataLoader(data, batch_size=8, shuffle=True)
    sample = next(iter(data_loader))
    print(sample[0].shape)
