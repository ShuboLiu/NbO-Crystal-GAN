import argparse
import os
import numpy as np
import pandas as pd
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import autograd
import torch.nn.init as init
from models import *


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def weights_init(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d : 
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0.0)

def noising(imgs):
    imgs = imgs.numpy()
    B = imgs.shape[0]
    mask = (imgs<0.01)
    a = np.random.normal(10**-3,10**-2.5,(B,1,30,3))
    noise = mask*abs(a)
    imgs_after_noising = imgs + noise
    imgs_after_noising = torch.tensor(imgs_after_noising)
    return imgs_after_noising	

def calc_gradient_penalty(netD, real_data, fake_data):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous().view(batch_size, 1, 30 , 3)
    alpha = alpha.cuda() if cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    feature, disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=501, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--d_lr', type=float, default=0.00005, help='adam: learning rate')
parser.add_argument('--q_lr', type=float, default=0.000025)
parser.add_argument('--g_lr', type=float, default=0.00005)
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=512, help='dimensionality of the latent space')
parser.add_argument('--model_save_dir', type = str, default = './model/')
parser.add_argument('--load_model', type = bool, default = False)
parser.add_argument('--load_generator', type = str)
parser.add_argument('--load_discriminator', type = str)
parser.add_argument('--load_q', type = str)
parser.add_argument('--constraint_epoch', type = int, default = 10000)
parser.add_argument('--gen_dir', type=str, default='./gen/')
parser.add_argument('--trainingdata', type=str, default='mgmno_2000.pickle')
parser.add_argument('--input_dim', type=str, default=512+28+1)
opt = parser.parse_args()
print(opt)

if not os.path.isdir(opt.gen_dir):
    os.makedirs(opt.gen_dir)
if not os.path.isdir(opt.model_save_dir):
    os.makedirs(opt.model_save_dir)

## Loss functions
adversarial_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()

## Initialize generator and discriminator
generator = Generator(opt)
discriminator = Discriminator(opt)
net_Q = QHead_(opt)
## Loss functions
adversarial_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()

## Initialize generator and discriminator
generator = Generator(opt)
discriminator = Discriminator(opt)
net_Q = QHead_(opt)

## Load data for training
## Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))
optimizer_Q = torch.optim.Adam(net_Q.parameters(), lr=opt.q_lr, betas=(opt.b1, opt.b2))

one = torch.FloatTensor([1])
mone = one * -1    

if cuda:
    one = one.cuda()
    mone = mone.cuda()
