import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import torch.utils.data as udata
import tqdm
import sys 
import pandas as pd
from PIL import Image
from utils import read_data_file, load_images_and_labels

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
'--config', '-c', default='configs/celebA_Young_Explainer.yaml')   
opt = parser.parse_args()

opt.n_epochs = 2
opt.batch_size = 8
opt.lr = 0.0002
opt.b1 = 0.5
opt.b2 = 0.999
opt.n_cpu = 8
opt.latent_dim = 100
opt.n_classes = 2
opt.img_size = 256
opt.channels = 3
opt.sample_interval = 400



img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class CelebaDataset(udata.TensorDataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, ns, attr_list, categories, transform=None):

        self.img_names = ns
        labels = np.zeros((ns.shape[0], categories), dtype=np.float32)
        for i, img_name in tqdm.tqdm(enumerate(ns)):
            try:
                labels[i] = attr_list[img_name]
            except:
                print(img_name)
  

        self.y = labels
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        label[np.where(label==-1)] = 0
        return img, label

    def __len__(self):
        return len(self.img_names)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

def convert_to_binary(y):
    for i in range(len(y)):
        if y[i] < 5:
            y[i] = 0
        else:
            y[i] = 1
    return y

# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
""" 
os.makedirs("data/mnist", exist_ok=True)
mnist_data = datasets.MNIST('data/mnist', download=True, 
                        transform=transforms.Compose(
                        [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                        ))
dataloader = DataLoader(mnist_data,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=8)
 """

# dataset
custom_transform = transforms.Compose([
    transforms.CenterCrop(opt.img_size),
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

image_label_dict = './output/classifier/CelebA-Young/explainer_input/list_attr_celeba_Young.txt'
try:
    categories, file_names_dict = read_data_file(image_label_dict)
except:
    print("Problem in reading input data file : ", image_label_dict)
    sys.exit()
data = np.asarray(list(file_names_dict.keys()))


# dataset
train_dataset = CelebaDataset(data,
                        file_names_dict,
                        1,
                    transform=custom_transform)

dataloader = udata.DataLoader(dataset=train_dataset,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=8)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        labels = torch.squeeze(convert_to_binary(labels)).int()
        #labels = torch.squeeze(labels).int()
        batch_size = imgs.shape[0]
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=opt.n_classes, batches_done=batches_done)