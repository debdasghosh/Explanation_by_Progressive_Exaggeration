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
    '--config', '-c', default='configs/celebA_Young_Classifier.yaml'
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
    # ============= Experiment Folder=============
    output_dir = os.path.join(config['log_dir'], config['name'])
    classifier_output_path = os.path.join(output_dir, 'classifier_output')
    try: os.makedirs(classifier_output_path)
    except: pass
    # ============= Experiment Parameters =============
    BATCH_SIZE = 32 #config['batch_size']
    EPOCHS = config['epochs']
    channels = config['num_channel']
    input_size = 128 #config['input_size'] 
    N_CLASSES = config['num_class'] 
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
    net = torch.load('./latest.pth')
    net = net.cuda()
    #net = net.to(device)
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

    
    for epoch in range(1):
        
        net.eval()

        names = np.empty([0])
        prediction_y = np.empty([0])
        true_y = np.empty([0])
        i = 0
        for data, target in trainLoader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = net(data)
            output = output[0:32]
            pred = (output>0.5).float()
            if i == 0:
                names = np.asarray(data.to("cpu"))
                prediction_y = np.asarray(pred.to("cpu"))
                true_y = np.asarray(target.to("cpu"))
            else:
                names = np.append(names, np.asarray(data.to("cpu")), axis= 0)
                prediction_y = np.append(prediction_y, np.asarray(pred.to("cpu")), axis=0)
                true_y = np.append(true_y, np.asarray(target.to("cpu")), axis= 0)
            print(i)
            i += 1
            if i == 100:
                break

        np.save(classifier_output_path + '/name_train1.npy', names)
        np.save(classifier_output_path + '/prediction_y_train1.npy', prediction_y)
        np.save(classifier_output_path + '/true_y_train1.npy', true_y)
        print('\nEpoch Train: {:.2f}'.format( epoch))
            

        names = np.empty([0])
        prediction_y = np.empty([0])
        true_y = np.empty([0])
        i = 0
        for data, target in testLoader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = net(data)
            output = output[0:32]
            pred = (output>0.5).float()
            
            if i == 0:
                names = np.asarray(data.to("cpu"))
                prediction_y = np.asarray(pred.to("cpu"))
                true_y = np.asarray(target.to("cpu"))
            else:
                names = np.append(names, np.asarray(data.to("cpu")), axis= 0)
                prediction_y = np.append(prediction_y, np.asarray(pred.to("cpu")), axis=0)
                true_y = np.append(true_y, np.asarray(target.to("cpu")), axis= 0)

            i += 1
            if i == 100:
                break
                
        np.save(classifier_output_path + '/name_test1.npy', names)
        np.save(classifier_output_path + '/prediction_y_test1.npy', prediction_y)
        np.save(classifier_output_path + '/true_y_test1.npy', true_y)
        print('\nEpoch Test: {:.2f}'.format( epoch))
            

    tb.close()


def get_num_correct(preds, labels):
    return preds.eq(labels).sum().item()




if __name__ == "__main__":
    run()