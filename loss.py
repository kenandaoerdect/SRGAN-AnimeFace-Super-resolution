import torch.nn as nn
import torch
from torchvision.models.vgg import vgg16


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, fake_img_hr, target_img_hr):
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(fake_img_hr), self.loss_network(target_img_hr))
        # img MSE Loss
        image_mse_loss = self.mse_loss(fake_img_hr, target_img_hr)
        return image_mse_loss + 0.006 * perception_loss


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.bec_loss = nn.BCELoss()

    def forward(self, logits_fake):
        # Adversarial Loss
        adversarial_loss = self.bec_loss(logits_fake, torch.ones_like(logits_fake))
        return 0.001 * adversarial_loss
