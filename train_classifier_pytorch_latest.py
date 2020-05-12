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
from torchvision import models
from torch.utils import data as udata
from torch.utils.data import DataLoader
import math
from torch.utils.tensorboard import SummaryWriter

from PIL import Image
from tqdm import tqdm

from torchsummary import summary

class CelebaDataset(udata.TensorDataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, ns, attr_list, categories, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = ns
        labels = np.zeros((ns.shape[0], categories), dtype=np.float32)
        for i, img_name in tqdm(enumerate(ns)):
            try:
                labels[i] = attr_list[img_name]
            except:
                print(img_name)
  

        self.y = labels
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        label[np.where(label==-1)] = 0
        return img, label

    def __len__(self):
        return len(self.img_names)



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
    BATCH_SIZE = 32 #config['batch_size']
    EPOCHS = config['epochs']
    channels = config['num_channel']
    input_size = 128 #config['input_size'] 
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
    torch.backends.cudnn.benchmark = True
    custom_transform = transforms.Compose([
       transforms.CenterCrop(input_size),
       transforms.Resize(input_size),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    tb = SummaryWriter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet169', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet161', pretrained=True)
    num_ftrs = net.classifier.in_features
    net.classifier = nn.Linear(num_ftrs, N_CLASSES)
    net = net.cuda()
    #net = net.to(device)
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    print(net)

    ns = data_train
    train_dataset = CelebaDataset(ns,
                            file_names_dict,
                            N_CLASSES,
                        img_dir=config['image_dir'],
                        transform=custom_transform)

    trainLoader = udata.DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=8)

    ns = data_test
    test_dataset = CelebaDataset(ns,
                            file_names_dict,
                            N_CLASSES,
                        img_dir=config['image_dir'],
                        transform=custom_transform)

    testLoader = udata.DataLoader(dataset=test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=8)

    for epoch in range(1, EPOCHS + 1):
        train( epoch, net, trainLoader, optimizer, tb)
            
        images, labels = next(iter(trainLoader))
        tb.add_graph(net, images)
        #torch.cuda.empty_cache()
        
        test( epoch, net, testLoader, optimizer, tb)
            
        torch.save(net, os.path.join('./', 'latest.pth'))
    tb.close()


def get_num_correct(preds, labels):
    return preds.eq(labels).sum().item()

def train( epoch, net, trainLoader, optimizer, tb ):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)

    total_loss = 0
    total_correct = 0

    for batch_idx, (data, target) in enumerate(trainLoader, 0):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        #loss = F.binary_cross_entropy(output, target)
        lossFn = nn.BCEWithLogitsLoss()
        loss = lossFn(output, target)
        loss.backward()
        optimizer.step()

        nProcessed += len(data)
        #pred = output.data.max(1)[1] 
        pred = (output>0.5).float()
        total_loss += loss.item()
        correct = get_num_correct(pred, target)
        total_correct += correct
        incorrect = pred.ne(target.data).cpu().sum()
        print('\nTrain Epoch: {:.2f} \tBatch ID: {:.2f} \tLoss: {:.6f}'.format( epoch, batch_idx, loss.data)) #loss.data[0]
        
    print('\nTrain Epoch: {:.2f} \tLoss: {:.6f}'.format( epoch, total_loss))
    tb.add_scalar('Loss', total_loss, epoch)
    tb.add_scalar('Number Correct', total_correct, epoch)


    


def test( epoch, net, testLoader, optimizer, tb):
    net.eval()
    test_loss = 0
    incorrect = 0

    total_loss = 0
    total_correct = 0

    for data, target in testLoader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        output = output[0:32]
        #test_loss += F.binary_cross_entropy(output, target)#.data[0]
        lossFn = nn.BCEWithLogitsLoss()
        loss = lossFn(output, target)
        test_loss = loss.item()
        #pred = output.data.max(1)[1] # get the index of the max log-probability
        pred = (output>0.5).float()
        incorrect += pred.ne(target.data).cpu().sum()
        total_loss += loss.item()
        correct = get_num_correct(pred, target)
        total_correct += correct

    
    print('\nEpoch: {:.2f}, Loss: {:.6f}'.format( epoch, total_loss))
        
    tb.add_scalar('Loss', total_loss, epoch)
    tb.add_scalar('Number Correct', total_correct, epoch)




if __name__ == "__main__":
    run()