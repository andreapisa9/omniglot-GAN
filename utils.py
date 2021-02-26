import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import numpy as np
import matplotlib
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import torchvision.utils as vutils
import matplotlib.animation as animation
from copy import deepcopy
from random import randrange

# creo l'antidominio utile per creare dataset da liste
######################################################################
def createAntidomain(domain, train):
    loaders_list=[[]]
    label_cat=[]
    img_cat=[]
    #scorro le mie 10 classi e creo una lista con tutti gli elementi per ogni classe
    if(train):
        loader = loader_single_class
    else:
        loader = loader_single_class_test
        
    for i,data_single_class in enumerate(loader):
        for j,(img,label) in enumerate(data_single_class):
            #print(label.view(-1))
            if label[0].item() != domain:
                label_cat.append(label)
                img_cat.append(img)
            
    loaders_list[0].append((img_cat, label_cat))
    #i tensori dentro la lista vengono tutti concatenati
    label_cat = torch.cat(label_cat, dim=0)
    img_cat = torch.cat(img_cat, dim=0)

    #create antidomain
    antiDomain = []
    for data_list in loaders_list:
        x = torch.stack(list(img_cat))
        y = torch.stack(list(label_cat))

        dataset = TensorDataset(x, y)
        # ogni volta che si itera il dataset, viene fatto a random
        #b_size è il batch_size per la modalità lenta anche qualora non ci sia il kshot attivato
        b_size=0
        if slow_mode == True and kShot_bool == False:
            b_size = batch_size
        else:
            b_size = kShot
        antiDomain.append(DataLoader(
                dataset,
                sampler=RandomSampler(dataset),
                batch_size=b_size
            )
        )
    return antiDomain[0]
#######################################################################
