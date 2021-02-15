import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size_0", type=int, default=8, help="size of each image dimension")
parser.add_argument("--img_size_1", type=int, default=3, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size_0, opt.img_size_1)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
     def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential( #input shape (1,8,3)
            nn.Conv2d(in_channels=1, #input height 
                      out_channels=16, #n_filter
                      kernel_size=1, #filter size
                      stride=1, #filter step
                      padding=0 #con2d出来的图片大小不变
                      ), #output shape (16,8,3)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1) #output shape (16,14,14)
            )
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 1, 1, 0), #output shape (32,8,3)
                                   nn.ReLU(),
                                   nn.MaxPool2d(1))
        self.out = nn.Linear(32*8*3, 8*3)
         
     def forward(self, x):
        x = imgs.view(opt.batch_size, 1, 8, 3).cuda().to(torch.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        img = output.reshape(opt.batch_size, 8, 3)
        based_on_ground = True
        if based_on_ground :
            ground = imgs[np.random.randint(0, opt.batch_size), :, :]
            ground = ground.cuda().to(torch.float32)
            img = img + ground
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
            nn.BatchNorm2d(512,0.8),nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (1,1), stride = 1, padding = 0),
            nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
            nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = (1,1), stride =1, padding =0)
            )
        self.softmax = nn.Softmax2d()

    def forward(self, img):
        img_flat = img.view(img.shape[0], 1, 8, 3)
        validity = self.model(img_flat)
        #validity = self.softmax(validity)
        return validity


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
train_data_raw = np.load(r"3.data_augmentation.npy")
count_train_data = 0
total_train_data = train_data_raw.shape[0]*train_data_raw.shape[2]
train_data = np.zeros(((total_train_data, 8, 3)))
for i in range(0, train_data_raw.shape[0]):
    for j in range(0, train_data_raw.shape[1]):
        train_data[count_train_data, :, :] = train_data_raw[i, j, :, :]
dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1)
    #alpha = alpha.expand(batch_size, int(real_samples.nelement()/batch_size)).contiguous().view(batch_size, 1, 8 , 3)
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    alpha = alpha.cuda() if cuda else alpha
    # Get random interpolation between real and fake samples
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    if cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = D(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() #* 10
    return gradient_penalty

# ----------
#  Training
# ----------


import numpy as np
np.set_printoptions(threshold=np.inf)

batches_done = 0
fake_imgs_save = train_data[0, :, :].reshape(((1, 8, 3)))
gloss = []; dloss = []; wloss = []
for epoch in range(opt.n_epochs):
    for i, (imgs) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        #print("real_img.shape=", real_imgs.shape, "fake_img.shape=", fake_imgs.shape)
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        Wasserstein_D = torch.mean(real_validity) - torch.mean(fake_validity)

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            gloss.append(g_loss.item())
            dloss.append(d_loss.item())
            wloss.append(Wasserstein_D)

            if batches_done % opt.sample_interval == 0:
                fake_imgs = fake_imgs.cpu()
                fake_imgs_raw = fake_imgs.detach().numpy()
                fake_imgs_save = np.concatenate((fake_imgs_save, fake_imgs_raw), axis=0)
                

            batches_done += opt.n_critic

print("We have the fake generation of shape", fake_imgs_save.shape)

## 保存Loss曲线
plt.subplot(3, 1, 1)
plt.plot(gloss, '-')
plt.ylabel('G_loss')
plt.subplot(3, 1, 2)
plt.plot(dloss, '-')
plt.ylabel('D_loss')
plt.subplot(3, 1, 3)
plt.plot(wloss, '-')
plt.ylabel('W_loss')
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) 
image_name="./loss_image/loss_image_"+now+r".jpg"
plt.savefig(image_name)
plt.show()

npy_name="./loss_image/fake_imgs_gen_"+now+r".npy"
np.save(npy_name, fake_imgs_save)

print("All Done")