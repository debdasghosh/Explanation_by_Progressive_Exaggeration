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
import losses_sngan as L
import utils_sngan as utils_sngan
import math
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter

def downsample(inputs):
    m = nn.AvgPool2d(kernel_size = 2, stride = 2, padding=0)
    return m(inputs)

def upsample(x):
    #h, w = x.size()[2:]
    #return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')
    m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    return m(x)


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

class Block(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False, num_classes=0):
        super(Block, self).__init__()

        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = in_ch != out_ch or downsample
        if h_ch is None:
            h_ch = out_ch
        self.num_classes = num_classes

        # Register layrs
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        if self.num_classes > 0:
            self.b1 = CategoricalConditionalBatchNorm2d(
                num_classes, in_ch)
            self.b2 = CategoricalConditionalBatchNorm2d(
                num_classes, h_ch)
        else:
            self.b1 = nn.BatchNorm2d(in_ch)
            self.b2 = nn.BatchNorm2d(h_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1)

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.tensor, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.tensor, gain=math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.tensor, gain=1)

    def forward(self, x, y=None, z=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y, z)

    def shortcut(self, x, **kwargs):
        if self.learnable_sc:
            if self.downsample:
                h = downsample(x)
            h = self.c_sc(h)
            return h
        else:
            return x

    def residual(self, x, y=None, z=None, **kwargs):

        if y is not None:
            h = self.b1(x, y, **kwargs)
        else:
            h = self.b1(x)
        h = self.activation(h)
        if self.downsample:
            h = downsample(h)
        h = self.c1(h)
        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h)
        return self.c2(self.activation(h))


class DecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, num_classes=0):
        super(DecoderBlock, self).__init__()

        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch or upsample
        if h_ch is None:
            h_ch = out_ch
        self.num_classes = num_classes

        # Register layrs
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        if self.num_classes > 0:
            self.b1 = CategoricalConditionalBatchNorm2d(
                num_classes, in_ch)
            self.b2 = CategoricalConditionalBatchNorm2d(
                num_classes, h_ch)
        else:
            self.b1 = nn.BatchNorm2d(in_ch)
            self.b2 = nn.BatchNorm2d(h_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1)

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.tensor, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.tensor, gain=math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.tensor, gain=1)

    def forward(self, x, y=None, z=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y, z)

    def shortcut(self, x, **kwargs):
        if self.learnable_sc:
            if self.upsample:
                h = upsample(x)
            h = self.c_sc(h)
            return h
        else:
            return x

    def residual(self, x, y=None, z=None, **kwargs):
        if y is not None:
            h = self.b1(x, y, **kwargs)
        else:
            h = self.b1(x)
        h = self.activation(h)
        if self.upsample:
            h = upsample(h)
        h = self.c1(h)
        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h)
        return self.c2(self.activation(h))


class Autoencoder(nn.Module):
    def __init__(self, num_classes=0, activation=F.relu):
        super(Autoencoder,self).__init__()
        self.activation=activation
        self.num_classes=num_classes
        '''
        if num_classes > 0:
            self.batchNorm = CategoricalConditionalBatchNorm2d(num_classes, 3)

        else:
            self.batchNorm = nn.BatchNorm2d(3)
        '''
        self.batchNorm = nn.BatchNorm2d(3)
        self.block1 = nn.Sequential(nn.ReLU(True),nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
        
        self.block2 = Block(64, 128, activation=activation, downsample=True, num_classes=num_classes)
        self.block3 = Block(128,256, activation=activation, downsample=True, num_classes=num_classes)
        self.block4 = Block(256,512, activation=activation, downsample=True, num_classes=num_classes)
        self.block5 = Block(512,1024, activation=activation, downsample=True, num_classes=num_classes)
        self.em_block = Block(1024,1024, activation=activation, downsample=True, num_classes=num_classes)

        self.decoder_block1 = DecoderBlock(1024, 1024, activation=activation, upsample=True, num_classes=num_classes)
        self.decoder_block2 = DecoderBlock(1024, 512, activation=activation, upsample=True, num_classes=num_classes)
        self.decoder_block3 = DecoderBlock(512,256, activation=activation, upsample=True, num_classes=num_classes)
        self.decoder_block4 = DecoderBlock(256,128, activation=activation, upsample=True, num_classes=num_classes)
        self.decoder_block5 = DecoderBlock(128,64, activation=activation, upsample=True, num_classes=num_classes)
        '''
        if num_classes > 0:
            self.decoder_batchNorm = CategoricalConditionalBatchNorm2d(num_classes, 3)
        else:
            self.decoder_batchNorm = nn.BatchNorm2d(3)
        '''
        self.decoder_batchNorm = nn.BatchNorm2d(64)
        self.decoder_block6 = nn.Sequential(         
        nn.ReLU(True),
        nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        nn.Tanh()
        )

        
    def forward(self,x,y=None):
        #print(x.shape)
        '''
        if y is not None:
            print(y.shape)
            x = self.batchNorm(x, y)
        else:
            x = self.batchNorm(x)
        '''
        x = self.batchNorm(x)
        x = self.block1(x)
        #print(x.shape)
        x = self.block2(x, y)
        #print(x.shape)
        x = self.block3(x, y)
        #print(x.shape)
        x = self.block4(x, y)
        #print(x.shape)
        x = self.block5(x, y)
        #print(x.shape)
        embedding = self.em_block(x, y)
        #print(x.shape)
        #print("decoder starts")
        x = self.decoder_block1(embedding, y)
        #print(x.shape)
        x = self.decoder_block2(x, y)
        #print(x.shape)
        x = self.decoder_block3(x, y)
        #print(x.shape)
        x = self.decoder_block4(x, y)
        #print(x.shape)
        x = self.decoder_block5(x, y)
        #print(x.shape)
        '''
        if y is not None:
            x = self.decoder_batchNorm(x, y)
        else:
            x = self.decoder_batchNorm(x)
        '''
        x = self.decoder_batchNorm(x)
        #print(x.shape)
        x = self.decoder_block6(x)
        #print(x.shape)
        return x, embedding


class D_Resblock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False, num_classes=0):
        super(D_Resblock, self).__init__()

        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = in_ch != out_ch or downsample
        if h_ch is None:
            h_ch = out_ch
        self.num_classes = num_classes

        # Register layrs
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1)

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.tensor, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.tensor, gain=math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.tensor, gain=1)

    def forward(self, x, y=None, z=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y, z)

    def shortcut(self, x, **kwargs):
        x = self.c_sc(x)
        if self.learnable_sc:
            if self.downsample:
                x = downsample(x)
            return x
        else:
            return x

    def residual(self, x, y=None, z=None, **kwargs):
        x = self.activation(x)
        x = self.c1(x)
        x = self.activation(x)
        x = self.c2(x)
        return downsample(x) 


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
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool2d(self.c2(h), 2)



class Discriminator(nn.Module):

    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(Discriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = D_Resblock(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = D_Resblock(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.block4 = D_Resblock(num_features * 4, num_features * 8,
                            activation=activation, downsample=True)
        self.block5 = D_Resblock(num_features * 8, num_features * 16,
                            activation=activation, downsample=True) # 1024



        self.l6 = nn.utils.spectral_norm(nn.Linear(num_features * 16, 1))
        if num_classes > 0:
            self.l_y = nn.utils.spectral_norm(
                nn.Embedding(num_classes, num_features * 16))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output



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

def get_noise(input_size, batch_size, device, num_classes, gen):

    noise = utils_sngan.sample_z(
        batch_size, input_size, device, 'normal'
    )
    pseudo_y = utils_sngan.sample_pseudo_labels(
            num_classes, batch_size, device
        )

    return noise, pseudo_y

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

def convert_to_binary(y):
    for i in range(len(y)):
        if y[i] < 5:
            y[i] = 0
        else:
            y[i] = 1
    return y


    

def main():

    writer = SummaryWriter('./SNGAN')
    NUMS_CLASS = 2
    BATCH_SIZE = 8
    input_size = 256
    EPOCHS = 3
    channels = 3
    image_label_dict = './output/classifier/CelebA-Young/explainer_input/list_attr_celeba_Young.txt'
    try:
        categories, file_names_dict = read_data_file(image_label_dict)
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
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # dataset
    train_dataset = CelebaDataset(data,
                            file_names_dict,
                            1,
                        transform=custom_transform)
    
    train_loader = udata.DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=8)
    
    num_classes = NUMS_CLASS
    gen = Autoencoder(num_classes, F.relu).cuda()
    distance = nn.MSELoss()
    #distance = L.GenLoss('hinge', False)
    #opt_gen = optim.Adam(gen.parameters(),weight_decay=1e-5)
    opt_gen = optim.Adam(gen.parameters(), 0.0002, (0.0, 0.9))
    #print(gen)

    dis = Discriminator(64, num_classes, F.relu).cuda()
    opt_dis = optim.Adam(dis.parameters(), 0.0002, (0.0, 0.9))
    #dis_criterion = L.DisLoss('hinge', False)
    #print(dis)

    counter = 1
    # Training loop
    for epoch in range(1, EPOCHS + 1):
    
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            target = torch.squeeze(convert_to_binary(target)).type(torch.long)
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            batch_size = data.shape[0]
            
            # Adversarial ground truths
            valid_gt = Variable(torch.cuda.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake_gt = Variable(torch.cuda.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # -----------------
            #  Train Generator
            # -----------------

            opt_gen.zero_grad()

            # Sample noise and labels as generator input
            #z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            #gen_labels = Variable(torch.cuda.LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
            z = data
            gen_labels = utils_sngan.sample_pseudo_labels(num_classes, batch_size, device)

            # Generate a batch of images
            gen_imgs, gen_embedding = gen(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = dis(gen_imgs, gen_labels)
            g_loss = distance(validity, valid_gt)

            g_loss.backward()
            opt_gen.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            opt_dis.zero_grad()

            # Loss for real images
            validity_real = dis(data, target)
            d_real_loss = distance(validity_real, valid_gt)

            # Loss for fake images
            validity_fake = dis(gen_imgs.detach(), gen_labels)
            d_fake_loss = distance(validity_fake, fake_gt)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            opt_dis.step()


            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, EPOCHS, batch_idx, len(train_loader), d_loss.item(), g_loss.item())
            )
            writer.add_scalar('Generator Loss', g_loss.item(), counter)
            writer.add_scalar('Discrinator Loss', d_loss.item(), counter)

            batches_done = epoch * len(train_loader) + batch_idx
            if batches_done % 400 == 0:
                n_row = NUMS_CLASS
                with torch.no_grad():
                    recon_imgs, recon_embedding = gen(data, target)
                    torchvision.utils.save_image(recon_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


if __name__ == '__main__':
    main()