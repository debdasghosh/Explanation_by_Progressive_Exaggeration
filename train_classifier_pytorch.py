import numpy as np
import pandas as pd
import sys 
import os
import pdb
import yaml
from utils import read_data_file, load_images_and_labels
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore")



import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils import data as udata
from torch.utils.data import DataLoader
import math
import densenet



def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--config', '-c', default='configs/celebA_DenseNet_Classifier.yaml'
)   
    args = parser.parse_args()
    # ============= Load config =============
    config_path = args.config
    config = yaml.load(open(config_path))
    print(config)
    # ============= Experiment Folder=============
    output_dir = os.path.join(config['log_dir'], config['name'])
    try: os.makedirs(output_dir)
    except: pass
    try: os.makedirs(os.path.join(output_dir, 'logs'))
    except: pass
    # ============= Experiment Parameters =============
    BATCH_SIZE = config['batch_size']
    EPOCHS = config['epochs']
    channels = config['num_channel']
    input_size = config['input_size'] 
    N_CLASSES = config['num_classes'] 
    ckpt_dir_continue = config['ckpt_dir_continue']    
    if ckpt_dir_continue == '':
        continue_train = False
    else:
        continue_train = True
    # ============= Data =============
    try:
        categories, file_names_dict = read_data_file(config['image_label_dict'])
    except:
        print("Problem in reading input data file : ", config['image_label_dict'])
        sys.exit()
    data_train = np.load(config['train'])
    data_test = np.load(config['test'])
    print("The classification categories are: ")
    print(categories)
    print('The size of the training set: ', data_train.shape[0])
    print('The size of the testing set: ', data_test.shape[0])
    fp = open(os.path.join(output_dir, 'setting.txt'), 'w')
    fp.write('config_file:'+str(config_path)+'\n')
    fp.close()


    torch.manual_seed(1)

    '''
    ns = data_train[0:int(BATCH_SIZE)*2]
    xs, ys = load_images_and_labels(ns, config['image_dir'],N_CLASSES, file_names_dict, input_size, channels, do_center_crop=True)
    tensor_x = torch.Tensor(xs) 
    tensor_y = torch.Tensor(ys)
    trainDataset = udata.TensorDataset(tensor_x,tensor_y) # create your datset
    trainLoader = udata.DataLoader(trainDataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True) # create your dataloader
    '''

    ns = data_test[0:int(BATCH_SIZE)*2]
    xs, ys = load_images_and_labels(ns, config['image_dir'],N_CLASSES, file_names_dict, input_size, channels, do_center_crop=True)
    tensor_x = torch.Tensor(xs) 
    tensor_y = torch.Tensor(ys)
    testDataset = udata.TensorDataset(tensor_x,tensor_y) # create your datset
    testLoader = udata.DataLoader(testDataset, batch_size=BATCH_SIZE, num_workers=1, pin_memory=True, shuffle=False) # create your dataloader

    trainLoader = testLoader

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=N_CLASSES)
    #net = densenet3.DenseNet3(40, N_CLASSES, 12, reduction=1.0, bottleneck=False, dropRate=0.0)
    net = net.cuda()
    #net = net.to(device)
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)

    for epoch in range(1, EPOCHS + 1):
            train( epoch, net, trainLoader, optimizer)
            test( epoch, net, testLoader, optimizer)
            torch.save(net, os.path.join('./', 'latest.pth'))



def train( epoch, net, trainLoader, optimizer ):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        #pred = output.data.max(1)[1] 
        pred = (output>0.5).float()
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data, err)) #loss.data[0]


def test( epoch, net, testLoader, optimizer):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.binary_cross_entropy(output, target)#.data[0]
        #pred = output.data.max(1)[1] # get the index of the max log-probability
        pred = (output>0.5).float()
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))




if __name__ == "__main__":
    run()