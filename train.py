import torch
import torch.nn as nn
from load_data import TrainDataset
from model import Generator, Discriminator
from loss import ContentLoss, AdversarialLoss
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import os


batchsize = 1
epochs = 100
learning_rate = 0.0001
train_data_path = 'data/train'
checkpoint_path = 'checkpoints'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
traindata = TrainDataset(train_data_path, transform)
traindata_loader = DataLoader(traindata, batch_size=batchsize, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netD = Discriminator().to(device)
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate)
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate)
bce = nn.BCELoss()
contentLoss = ContentLoss().to(device)
adversarialLoss = AdversarialLoss()
# print(netG)
# print(netD)

if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

torch.save(netG, 'netG-epoch_000.pth')
for epoch in range(1, epochs+1):
    for idx, (lr, hr) in enumerate(traindata_loader):
        lr = lr.to(device)
        hr = hr.to(device)

        # 更新判别器
        netD.zero_grad()
        logits_fake = netD(netG(lr).detach())
        logits_real = netD(hr)
        # Lable smoothing
        real = torch.tensor(torch.rand(logits_real.size())*0.25 + 0.85).to(device)
        fake = torch.tensor(torch.rand(logits_fake.size())*0.15).to(device)
        d_loss = bce(logits_real, real) + bce(logits_fake, fake)
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        # 更新生成器
        netG.zero_grad()
        g_loss = contentLoss(netG(lr), hr) + adversarialLoss(logits_fake)
        g_loss.backward()
        optimizerG.step()

        print('Epoch:[%d/%d]\tStep:[%d/%d]\tD_loss:%6f\tG_loss:%6f'%
              (epoch, epochs, idx, len(traindata_loader), d_loss.item(), g_loss.item()))

        if epoch % 10 == 0:
            torch.save(netG, 'netG-epoch_%03d.pth' % epoch)
            # torch.save(netD, 'netD-epoch_%03d.pth' % epoch)