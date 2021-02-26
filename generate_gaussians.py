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

from matplotlib import pyplot as plt
import os
import ftplib
import argparse

import scipy.stats as stats
import math

from models import Discriminator, Generator, MLP, MLP_cls, MLP_mean_std
from load_custom_MNIST import customMNIST
from trainer import Trainer

from DCGAN_MLP import get_args, create_filtered_dataloader

def save_gaussian(z, name):
    normal_mean = torch.mean(z).cpu().numpy()
    normal_var = torch.var(z).cpu().numpy()

    print("{} MEAN: {}".format(name, normal_mean))
    print("{} VAR: {}".format(name, normal_var))

    sigma = math.sqrt(normal_var)
    x = np.linspace(normal_mean - 3*sigma, normal_mean + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, normal_mean, sigma))
    plt.show()
    plt.savefig('inference/{}.png'.format(name))
    plt.close()


def save_multiple_gaussians(z_dict):

    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]

    for n in range(10):
        z = z_dict["{}".format(n)]
        normal_mean = torch.mean(z).cpu().numpy()
        normal_var = torch.var(z).cpu().numpy()

        sigma = math.sqrt(normal_var)
        x = np.linspace(normal_mean - 3*sigma, normal_mean + 3*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, normal_mean, sigma), color=colors[n], label='{}'.format(n))
    
    plt.legend(loc="upper right")
    plt.show()
    plt.savefig('inference/multiple_gauss.png')
    plt.close()

def save_multiple_tensors(z_dict, name = "multiple_tensors"):
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]

    for n in range(10):
        z = z_dict["{}".format(n)].squeeze()
        z = torch.mean(z, dim= 0).cpu().numpy()
        plt.plot(z, color=colors[n], label='{}'.format(n))
    

    plt.legend(loc="upper right")
    plt.show()
    plt.savefig('inference/{}.png'.format(name))
    plt.close()

args = get_args()
args.mode = 'mlp_mean_std'

def save_multiple_heatmaps(z_dict, name = "multiple_heatmaps"):

    cmap = plt.get_cmap('hot')
    h_map = torch.mean(z_dict["0"].squeeze(), dim= 0).unsqueeze(0)
    
    for i in range(1,10):
        h_map_b = torch.mean(z_dict["{}".format(i)].squeeze(), dim= 0).unsqueeze(0)
        h_map = torch.cat((h_map,h_map_b), dim = 0)
    
    h_map = h_map.cpu().numpy()
    plt.imshow(h_map, cmap='hot', interpolation='nearest')
    plt.savefig('inference/{}.png'.format(name))
    plt.close()

args = get_args()
args.mode = 'mlp_mean_std'

NUM_LABELS = args.num_labels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_G = Generator().cuda()
model_D = Discriminator().cuda()
if args.mode == 'standard' or args.mode == 'noise' or args.mode == 'concat':
    model_MLP_cls = MLP_cls().cuda()

if args.mode == 'mlp_mean_std':
    model_MLP_cls = MLP_mean_std().cuda()

model_G = torch.load('models_KL/20000_G.pth')
#model_G.eval()
model_D = torch.load('models_KL/20000_D.pth')
model_MLP = torch.load('models_KL/20000_MLP.pth')
model_MLP_cls = torch.load('models_KL/20000_MLP_cls.pth')

# Setup Adam optimizers for both G and D
optimizer_D = optim.Adam(model_D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizer_G = optim.Adam(model_G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizer_MLP = optim.Adam(model_MLP.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizer_MLP_cls = optim.Adam(model_MLP_cls.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

models = {
    "G" : model_G,
    "D" : model_D,
    "MLP" : model_MLP,
    "MLP_cls" : model_MLP_cls
}

optimizers = {
    "G" : optimizer_G,
    "D" : optimizer_D,
    "MLP" : optimizer_MLP,
    "MLP_cls" : optimizer_MLP_cls
}

transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor()#,
    #transforms.Normalize((0.1307,), (0.3081,))
    ])

# load N-1 class loaders
loader_single_class_test = [create_filtered_dataloader(args, customMNIST(root = 'data',
                    label = i, train = False, transform=transform)) 
                    for i in range(NUM_LABELS)]

loader_masked_class_test = [create_filtered_dataloader(args, customMNIST(root = 'data',
                    label = i, train = False, transform=transform, mask_mode=True)) 
                    for i in range(NUM_LABELS)]

test_dataset = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    sampler=RandomSampler(test_dataset),
    batch_size=args.batch_size
)

fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)
trainer = Trainer(args, models, optimizers, device, fixed_noise)

noise = torch.randn(args.batch_size, args.nz, 1, 1).cuda()


save_gaussian(noise, "gaussian_normal")

noise_dict = {}
mean_dict = {}
std_dict = {}

for n in range(NUM_LABELS):
    weights_before_D = deepcopy(models["D"].state_dict()) # save snapshot before evaluation
    weights_before_G = deepcopy(models["G"].state_dict()) # save snapshot before evaluation
    weights_before_MLP = deepcopy(models["MLP"].state_dict()) # save snapshot before evaluation
    weights_before_MLP_cls = deepcopy(models["MLP_cls"].state_dict())

    single_loader_test = loader_single_class_test[n] #selezioni il loader della classe n
    masked_loader_test = loader_masked_class_test[n]
    # aggiorno la rete su una singola classe
    err_D, err_MLP = trainer.train_GAN_on_task(single_loader_test, masked_loader_test, test_loader)
    #err_G = trainer.train_G()

    with torch.no_grad():
        out_imgs_fake = trainer.generate_sample().detach().cpu()
        
    out_z = trainer.get_z()
    noise_dict["{}".format(n)] = out_z

    mean, std = trainer.get_mean_std()
    mean_dict["{}".format(n)] = mean
    std_dict["{}".format(n)] = std

    save_gaussian(out_z, "gaussian_{}".format(n))

    # salvi le immagini generate
    print("SAVING IMGS {}".format(n))
    save_image(out_imgs_fake,"inference/" + f"{n}.png") #aggiungi percorso: "path/iterazione_classe.png" es "pippo/20000_3.png"
        
    # ripristini il modello prima dell'aggiornamento su una singola classe
    models["D"].load_state_dict(weights_before_D) # restore from snapshot   
    models["G"].load_state_dict(weights_before_G)
    models["MLP"].load_state_dict(weights_before_MLP)
    models["MLP_cls"].load_state_dict(weights_before_MLP_cls)

#noise_dict["10"] = noise
save_multiple_gaussians(noise_dict)
save_multiple_tensors(noise_dict)
save_multiple_tensors(mean_dict, name = "mean_tensors")
save_multiple_tensors(std_dict, name = "std_tensors")
save_multiple_heatmaps(noise_dict)
save_multiple_heatmaps(mean_dict, name = "mean_heatmaps")
save_multiple_heatmaps(std_dict, name = "std_heatmaps")