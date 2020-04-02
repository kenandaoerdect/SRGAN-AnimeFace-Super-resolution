import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from load_data import TrainDataset
import torchvision.utils as vutils


test_img_path = 'data/test/'
checkpoint_path = 'checkpoints/netG-epoch_100.pth'
save_img_path = os.path.join('result', 'fake_hr_%s'%checkpoint_path.split('.')[0][-9:])
if not os.path.exists(save_img_path):
	os.makedirs(save_img_path)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
test_data = TrainDataset(test_img_path,)
testloader = DataLoader(test_data, batch_size=1, shuffle=False)


netG = torch.load(checkpoint_path, map_location=torch.device('cpu'))

for idx, (lr, hr) in enumerate(testloader):
	fake_hr = netG(lr)
	vutils.save_image(fake_hr.data, 
					  '%s/%03d.png'%(save_img_path, idx),
					  normalize=True)
	print(idx)