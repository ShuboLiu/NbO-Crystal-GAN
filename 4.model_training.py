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
import torch.nn.init as init

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
parser.add_argument("--img_size_0", type=int, default=4, help="size of each image dimension")
parser.add_argument("--img_size_1", type=int, default=3, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size_0, opt.img_size_1)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def weights_init(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d : 
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0.0)

def noising(imgs):
    imgs = imgs.cpu().numpy().reshape((((opt.batch_size, 1, opt.img_size_0, opt.img_size_1))))
    batch_size = imgs.shape[0]
    mask = (imgs<0.01)
    a = np.random.normal(10**-3,10**-2.5,(batch_size,1,opt.img_size_0, opt.img_size_1))
    noise = mask*abs(a)
    imgs_after_noising = imgs + noise
    imgs_after_noising = torch.tensor(imgs_after_noising)
    return imgs_after_noising	

def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.

    return Variable(FloatTensor(y_cat))

def count_element(label):
    n_x  = (label==1).sum(dim=1)
    return n_x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + 12 +1
        self.input_dim = input_dim # 200+6+1

        self.l1 = nn.Sequential(nn.Linear(input_dim, 128*4),nn.ReLU(True))
        self.map1 = nn.Sequential(nn.ConvTranspose2d(128,256,(1,3),stride = 1,padding=0),nn.BatchNorm2d(256,0.8),nn.ReLU(True)) #(28,3)
        self.map2 = nn.Sequential(nn.ConvTranspose2d(256,512,(1,1),stride = 1,padding=0),nn.BatchNorm2d(512,0.8),nn.ReLU(True)) #(28,3)
        self.map3 = nn.Sequential(nn.ConvTranspose2d(512,256,(1,1),stride = 1,padding=0),nn.BatchNorm2d(256,0.8),nn.ReLU(True)) #(28,3)
        self.map4 = nn.Sequential(nn.ConvTranspose2d(256,1,(1,1),stride=1,padding=0)) #(28,3)
        self.cellmap = nn.Sequential(nn.Linear(84,30),nn.BatchNorm1d(30),nn.ReLU(True),nn.Linear(30,6),nn.Sigmoid())

        self.sigmoid = nn.Sigmoid()

    def forward(self, z, fake_labels_Nb, fake_labels_o, natoms_fake):
        x = imgs.view(opt.batch_size, 1, opt.img_size_0, opt.img_size_1).cuda().to(torch.float32)
        #cell_param = np.random.randint(0, 3, opt.batch_size)
        #c1 = x[:, :, 2:5, :]
        #c2 = x[:, :, 5:8, :]
        #print("c1.shape=", c1.shape,"c2.shape=", c2.shape)
        #print("z.shape=", z.shape,"fake_labels_Nb.shape=", fake_labels_Nb.shape,
        #    "fake_labels_o.shape=", fake_labels_o.shape,"natoms_fake.shape=", natoms_fake.shape)

        gen_input = torch.cat((z, fake_labels_Nb, fake_labels_o, natoms_fake), -1)
        #print(gen_input.shape)
        h = self.l1(gen_input)
        h = h.view(h.shape[0], 128, opt.img_size_0, 1)
        h = self.map1(h)
        h = self.map2(h)
        h = self.map3(h)
        h = self.map4(h)

        h_flatten = h.view(h.shape[0],-1)
        img = h_flatten.reshape(opt.batch_size, opt.img_size_0, opt.img_size_1)
        based_on_ground = False
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
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
            nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
            nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = (1,1), stride =1, padding =0)
            )
        self.softmax = nn.Softmax2d()

    def forward(self, img):
        img_flat = img.view(img.shape[0], 1, opt.img_size_0, opt.img_size_1)
        validity = self.model(img_flat)
        #validity = self.softmax(validity) ##实践证明，用了Softmax之后Loss直接起飞
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
total_train_data = train_data_raw.shape[0]*train_data_raw.shape[1]
#train_data = np.zeros(((total_train_data, opt.img_size_0, opt.img_size_1)))
train_data = np.empty((total_train_data, opt.img_size_0, opt.img_size_1), dtype = float)
for i in range(0, train_data_raw.shape[0]):
    for j in range(0, train_data_raw.shape[1]):
        train_data[count_train_data, :, :] = train_data_raw[i, j, :, :]
        count_train_data += 1
if count_train_data == total_train_data :
    print("Data size fits")
else:
    print("Danger!")
    print("count_train_data=",count_train_data, "total_train_data=", total_train_data)
    os._exit(0)

print(train_data)
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

## Load model or Initialize
load_model = False
if load_model:
    generator.load_state_dict(torch.load(opt.load_generator))
    discriminator.load_state_dict(torch.load(opt.load_discriminator))
    print("load model ! ", opt.load_generator, opt.load_discriminator, opt.load_q)
else:
    generator.apply(weights_init)
    print("generator weights are initialized")
    discriminator.apply(weights_init)
    print("discriminator weights are initialized")

batches_done = 0
fake_imgs_save = train_data[0, :, :].reshape(((1, opt.img_size_0, opt.img_size_1)))
gloss = []; dloss = []; wloss = []
for epoch in range(opt.n_epochs):
    for i, (imgs) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        real_imgs_noise = noising(real_imgs)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = autograd.Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        if cuda :
            z = z.cuda()
        
        Nb_label = imgs[:, 2:5, :]
        O_label = imgs[:, 5:8, :]
        n_Nb = count_element(Nb_label).reshape(opt.batch_size, -1)
        n_o = count_element(O_label).reshape(opt.batch_size, -1)
        natoms = n_Nb + n_o
        n_Nb = n_Nb -1
        n_o = n_o -1
        real_imgs = autograd.Variable(real_imgs.type(FloatTensor))
        real_imgs_noise = autograd.Variable(real_imgs_noise.type(FloatTensor))
        real_labels_Nb = autograd.Variable(n_Nb.type(LongTensor))			
        real_labels_o = autograd.Variable(n_o.type(LongTensor))
        Nb_label = autograd.Variable(Nb_label.type(LongTensor))
        O_label = autograd.Variable(O_label.type(LongTensor))
        cell_label = autograd.Variable((natoms.type(FloatTensor))/(6.0)).unsqueeze(-1)

        fake_labels_Nb_int = np.random.randint(0, 3, opt.batch_size)
        fake_labels_Nb = to_categorical(fake_labels_Nb_int,num_columns = 6)
        fake_labels_o_int = np.random.randint(0,3,opt.batch_size)
        fake_labels_o = to_categorical(fake_labels_o_int, num_columns = 6)
        natoms_fake = fake_labels_Nb_int + fake_labels_o_int + 3
        natoms_fake = Variable(FloatTensor(natoms_fake)/(6.0)).unsqueeze(-1)

        # Generate a batch of images
        fake_imgs = generator(z, fake_labels_Nb, fake_labels_o, natoms_fake)
        fake_imgs = autograd.Variable(fake_imgs)

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
            fake_imgs = generator(z, fake_labels_Nb, fake_labels_o, natoms_fake)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [W loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), Wasserstein_D)
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

npy_name="./loss_image/fake_imgs_gen.npy"
np.save(npy_name, fake_imgs_save)

print("All Done")