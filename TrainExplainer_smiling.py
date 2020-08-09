import argparse
import datetime
import json
import os
import shutil

import numpy as np
import torch
import torch.utils.data as udata
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import torch.nn as nn

import sys 
import pandas as pd
from PIL import Image
#from tqdm import tqdm
from torchsummary import summary
import pdb
import yaml
from utils import read_data_file, load_images_and_labels
from torch.autograd import Variable
import utils_sngan as utils_sngan
import math
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter


# # Utility Functions


def downsample(inputs):
    #m = nn.AvgPool2d(kernel_size = 2, stride = 2, padding=0)
    #return m(inputs)
    return F.avg_pool2d(inputs, 2)


# In[3]:


def upsample(x):
    #h, w = x.size()[2:]
    #return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')
    m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    return m(x)


# In[4]:


class ConditionalBatchNorm2d(nn.BatchNorm2d):

    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias


# In[5]:


class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)
        return super(CategoricalConditionalBatchNorm2d, self).forward(input, weight, bias)


# In[6]:


def save_checkpoints(results_root, n_iter, count, gen, opt_gen, dis, opt_dis):
    """Save checkpoints.

    Args:
        args (argparse object)
        n_iter (int)
        gen (nn.Module)
        opt_gen (torch.optim)
        dis (nn.Module)
        opt_dis (torch.optim)

    """

    gen_dst = os.path.join(
        results_root,
        'gen_{}_iter_{:07d}.pth.tar'.format(count, n_iter)
    )
    torch.save({
        'model': gen.state_dict(), 'opt': opt_gen.state_dict(),
    }, gen_dst)
    shutil.copy(gen_dst, os.path.join(results_root, 'gen_latest.pth.tar'))
    dis_dst = os.path.join(
        results_root,
        'dis_{}_iter_{:07d}.pth.tar'.format(count, n_iter)
    )
    torch.save({
        'model': dis.state_dict(), 'opt': opt_dis.state_dict(),
    }, dis_dst)
    shutil.copy(dis_dst, os.path.join(results_root, 'dis_latest.pth.tar'))


# In[7]:


class img_norm(object):
    def __call__(self,img):
        img = img / 255.0 
        img = img - 0.5
        img = img * 2.0
        return  img  


# In[8]:


def loss_hinge_dis(dis_fake, dis_real):
    loss = torch.nn.functional.relu(1.0 - dis_real).mean() +            torch.nn.functional.relu(1.0 + dis_fake).mean()
    return loss


# In[9]:


def loss_hinge_gen(dis_fake):
    loss = -dis_fake.mean()
    return loss


# # Autoencoder

# In[10]:


class Block(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1,
                 activation=F.relu, num_classes=0):
        super(Block, self).__init__()

        self.activation = activation
        self.num_classes = num_classes

        # Register layrs
        self.c1 = nn.Conv2d(in_ch, out_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(out_ch, out_ch, ksize, 1, pad)
        if self.num_classes > 0:
            self.b1 = CategoricalConditionalBatchNorm2d(
                num_classes, in_ch)
            self.b2 = CategoricalConditionalBatchNorm2d(
                num_classes, out_ch)
        else:
            self.b1 = nn.BatchNorm2d(in_ch)
            self.b2 = nn.BatchNorm2d(out_ch)
        self.c_sc = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, gain=math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data, gain=1)

    def forward(self, x, y=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y)

    def shortcut(self, x, **kwargs):
        h = downsample(x)
        h = self.c_sc(h)
        return h
        
    def residual(self, x, y=None, **kwargs):

        if y is not None:
            h = self.b1(x, y, **kwargs)
        else:
            h = self.b1(x)
        h = self.activation(h)
        h = downsample(h)
        h = self.c1(h)
        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h


# In[11]:


class DecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1,
                 activation=F.relu, num_classes=0):
        super(DecoderBlock, self).__init__()

        self.activation = activation
        self.num_classes = num_classes

        # Register layrs
        self.c1 = nn.Conv2d(in_ch, out_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(out_ch, out_ch, ksize, 1, pad)
        if self.num_classes > 0:
            self.b1 = CategoricalConditionalBatchNorm2d(
                num_classes, in_ch)
            self.b2 = CategoricalConditionalBatchNorm2d(
                num_classes, out_ch)
        else:
            self.b1 = nn.BatchNorm2d(in_ch)
            self.b2 = nn.BatchNorm2d(out_ch)
        self.c_sc = nn.Conv2d(in_ch, out_ch, 1)
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, gain=math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data, gain=1)

    def forward(self, x, y=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y)

    def shortcut(self, x, **kwargs):
        h = upsample(x)
        h = self.c_sc(h)
        return h
        

    def residual(self, x, y=None, **kwargs):
        if y is not None:
            h = self.b1(x, y, **kwargs)
        else:
            h = self.b1(x)
        h = self.activation(h)
        h = upsample(h)
        h = self.c1(h)
        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h)
        return self.c2(self.activation(h))


# In[12]:


class Autoencoder(nn.Module):
    def __init__(self, num_classes=0, activation=F.relu):
        super(Autoencoder,self).__init__()
        self.activation=activation
        self.num_classes=num_classes

        self.batchNorm = nn.BatchNorm2d(3)
        self.block1_a = nn.ReLU(True)
        self.block1_b = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.block2 = Block(64, 128, activation=activation, num_classes=num_classes)
        self.block3 = Block(128,256, activation=activation, num_classes=num_classes)
        self.block4 = Block(256,512, activation=activation, num_classes=num_classes)
        self.block5 = Block(512,1024, activation=activation, num_classes=num_classes)
        self.em_block = Block(1024,1024, activation=activation, num_classes=num_classes)

        self.decoder_block1 = DecoderBlock(1024, 1024, activation=activation, num_classes=num_classes)
        self.decoder_block2 = DecoderBlock(1024, 512, activation=activation, num_classes=num_classes)
        self.decoder_block3 = DecoderBlock(512,256, activation=activation, num_classes=num_classes)
        self.decoder_block4 = DecoderBlock(256,128, activation=activation, num_classes=num_classes)
        self.decoder_block5 = DecoderBlock(128,64, activation=activation, num_classes=num_classes)

        self.decoder_batchNorm = nn.BatchNorm2d(64)
        self.decoder_block6_a = nn.ReLU(True)
        self.decoder_block6_b = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.decoder_block6_c = nn.Tanh()

        
    def forward(self,x,y=None):

        x = self.batchNorm(x)
        x = self.block1_a(x)
        x = self.block1_b(x)
        x = self.block2(x, y)
        x = self.block3(x, y)
        x = self.block4(x, y)
        x = self.block5(x, y)
        embedding = self.em_block(x, y)

        x = self.decoder_block1(embedding, y)
        x = self.decoder_block2(x, y)
        x = self.decoder_block3(x, y)
        x = self.decoder_block4(x, y)
        x = self.decoder_block5(x, y)
        x = self.decoder_batchNorm(x)
        x = self.decoder_block6_a(x)
        x = self.decoder_block6_b(x)
        x = self.decoder_block6_c(x)
        
        return x, embedding


# # Discriminator

# In[13]:


class D_Resblock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1,
                 activation=F.relu, num_classes=0, downsample=True):
        super(D_Resblock, self).__init__()

        self.activation = activation
        self.num_classes = num_classes
        self.downsample = downsample

        # Register layrs
        self.c1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, gain=math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data, gain=1)

    def forward(self, x, y=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y)

    def shortcut(self, x, **kwargs):
        x = self.c_sc(x)
        if self.downsample:
            x = downsample(x)
        return x
        

    def residual(self, x, y=None, z=None, **kwargs):
        x = self.activation(x)
        x = self.c1(x)
        x = self.activation(x)
        x = self.c2(x)
        if self.downsample:
            x = downsample(x)
        return x


# In[14]:


class OptimizedBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data, 1)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool2d(self.c2(h), 2)


# In[15]:


class Discriminator(nn.Module):

    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(Discriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = D_Resblock(num_features, num_features * 2,
                            activation=activation)
        self.block3 = D_Resblock(num_features * 2, num_features * 4,
                            activation=activation)
        self.block4 = D_Resblock(num_features * 4, num_features * 8,
                            activation=activation)
        self.block5 = D_Resblock(num_features * 8, num_features * 16,
                            activation=activation) # 1024
        self.block6 = D_Resblock(num_features * 16, num_features * 16,
                            activation=activation, downsample=False)

        self.l6 = nn.utils.spectral_norm(nn.Linear(num_features * 16, 1))
        if num_classes > 0:
            self.l_y = nn.utils.spectral_norm(
                nn.Embedding(num_classes, num_features * 16))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data, 1)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data, 1)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)

        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
            
        return output


# # Dataloader Class

# In[16]:


class CelebaDataset(udata.TensorDataset): #torch.utils.data.Dataset
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


# # Config

# In[17]:


writer = SummaryWriter('/pylon5/ac5616p/debdas/Explanation/tensorboard')
NUMS_CLASS = 2
BATCH_SIZE = 8
input_size = 128
EPOCHS = 3
channels = 3
lambda_GAN = 1
lambda_cyc = 100
image_dir = '/pghbio/dbmi/batmanlab/singla/MICCAI_2019/GAN_Interpretability/data/celebA/images/'
image_label_dict = '/pylon5/ac5616p/debdas/Explanation/data/CelebA/Smiling_binary_classification.txt'
try:
    categories, file_names_dict = read_data_file(image_label_dict, image_dir)
except:
    print("Problem in reading input data file : ", image_label_dict)
    sys.exit()
data = np.asarray(list(file_names_dict.keys()))

# CUDA setting
if not torch.cuda.is_available():
    raise ValueError("Should buy GPU!")
torch.manual_seed(46)
torch.cuda.manual_seed_all(46)
device = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.benchmark = True

# dataset
custom_transform = transforms.Compose([
   transforms.CenterCrop(input_size),
   transforms.Resize(input_size),
   transforms.ToTensor(),
   transforms.Normalize((0.5,), (0.5,)),
   #img_norm()
])

# dataset
train_dataset = CelebaDataset(data,
                        file_names_dict,
                        1,
                        transform=custom_transform
                    )

train_loader = udata.DataLoader(dataset=train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=0)

num_classes = NUMS_CLASS
gen = Autoencoder(num_classes, F.relu).cuda()
#gen = Autoencoder(num_classes, 3).cuda()


l2_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
opt_gen = optim.Adam(gen.parameters(), 0.0002, (0.0, 0.9))
#print(gen)

dis = Discriminator(64, num_classes, F.relu).cuda()
#dis = Discriminator(num_classes, 3).cuda()
opt_dis = optim.Adam(dis.parameters(), 0.0002, (0.0, 0.9))
#print(dis)
torch.autograd.set_detect_anomaly(True)
counter = 1
g_loss = 0
d_loss = 0


# # Training

# In[ ]:


# Training loop
for epoch in range(1, EPOCHS + 1):

    for batch_idx, (data, target) in enumerate(train_loader, 0):
        data, target = data.cuda(), target.type(torch.long).cuda()
        data, target = Variable(data), Variable(target)
        batch_size = data.shape[0]
        #print(data[0])
        x_source = data
        y_source = torch.squeeze(target)
        y_target = utils_sngan.sample_pseudo_labels(num_classes, data.shape[0], device)

        # Generate Fake Target Image
        fake_target_img, fake_target_img_embedding = gen(x_source, y_target) #

        # Train Discriminator
        opt_dis.zero_grad()
        real_source_logits = dis(x_source, y_source) #
        fake_target_logits = dis(fake_target_img.clone().detach(), y_target) # 
        D_loss_GAN = loss_hinge_dis(real_source_logits, fake_target_logits)
        D_loss = (D_loss_GAN * 1)
        d_loss = D_loss.item()

        D_loss.backward() #retain_graph=True
        opt_dis.step()

        opt_gen.zero_grad()

        #if (batch_idx+1) % 5 == 0:
        # Train Generator
        fake_source_img, fake_source_img_embedding = gen(fake_target_img, y_source) #
        fake_source_recons_img, x_source_img_embedding = gen(x_source, y_source) #

        G_loss_GAN = loss_hinge_gen(fake_target_logits.detach()) #
        G_loss_cyc = l1_loss(x_source, fake_target_img) 
        G_loss_rec = l2_loss(x_source_img_embedding, fake_source_img_embedding)
        G_loss = (G_loss_GAN * 1) + (G_loss_cyc * 10) + (G_loss_rec * lambda_cyc)    
    
        G_loss.backward() #retain_graph=True
        opt_gen.step()
        g_loss = G_loss.item()
        

        if counter % 400 == 0:
            writer.add_images('real_img', x_source, counter)
            writer.add_images('fake_target_img', fake_target_img, counter)
            writer.add_images('fake_source_img', fake_source_img, counter)
            writer.add_images('fake_source_recons_img', fake_source_recons_img, counter)

            writer.add_scalar('loss_g', g_loss, counter)
            writer.add_scalar('loss_d', d_loss, counter)
            writer.add_scalar('loss_g_GAN', G_loss_GAN, counter)
            writer.add_scalar('loss_d_GAN', D_loss_GAN, counter)
            writer.add_scalar('G_loss_cyc', G_loss_cyc, counter)
            writer.add_scalar('G_loss_rec', G_loss_rec, counter)

        batches_done = epoch * len(train_loader) + batch_idx
        if batches_done % 400 == 0:
            
            print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, EPOCHS, batch_idx, len(train_loader), d_loss, g_loss)
            )

            with torch.no_grad():
                recon_imgs, recon_embedding = gen(x_source, y_target)
                torchvision.utils.save_image(recon_imgs.data, "/pylon5/ac5616p/debdas/Explanation/images/%d.png" % batches_done, nrow=2, normalize=True)

        counter += 1

    # Save checkpoints!
    save_checkpoints(
        '/pylon5/ac5616p/debdas/Explanation/tensorboard', epoch, epoch // EPOCHS,
        gen, opt_gen, dis, opt_dis
    )

