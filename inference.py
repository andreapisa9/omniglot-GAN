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

from models import Discriminator, Generator, MLP, MLP_cls
from load_custom_Omniglot import customOmniglot
from utils import createAntidomain
from trainer import Trainer

from DCGAN_MLP import get_args, create_filtered_dataloader

args = get_args()

NUM_LABELS = args.num_labels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_G = Generator().cuda()
model_D = Discriminator().cuda()
model_MLP = MLP().cuda()
model_MLP_cls = MLP_cls().cuda()

model_G = torch.load('models/G.pth')
#model_G.eval()
model_D = torch.load('models/D.pth')
model_MLP = torch.load('models/MLP.pth')
model_MLP_cls = torch.load('models/MLP_cls.pth')


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

random_noise = torch.randn(64, args.nz + 100, 1, 1, device=device)
out_imgs = model_G(random_noise).detach().cpu()
save_image(out_imgs,"inference/no_meta_inference.png")

transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor()#,
    #transforms.Normalize((0.1307,), (0.3081,))
    ])

# load N-1 class loaders
loader_single_class_test = [create_filtered_dataloader(args, customOmniglot(root = 'data',
                    label = i, train = False, transform=transform)) 
                    for i in range(NUM_LABELS)]

loader_masked_class_test = [create_filtered_dataloader(args, customOmniglot(root = 'data',
                    label = i, train = False, transform=transform, mask_mode=True)) 
                    for i in range(NUM_LABELS)]

test_dataset = datasets.Omniglot(
    root='data',
    background=False,
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
        
    # salvi le immagini generate
    print("SAVING IMGS {}".format(n))
    save_image(out_imgs_fake,"inference/" + f"{n}.png") #aggiungi percorso: "path/iterazione_classe.png" es "pippo/20000_3.png"
        
    # ripristini il modello prima dell'aggiornamento su una singola classe
    models["D"].load_state_dict(weights_before_D) # restore from snapshot   
    models["G"].load_state_dict(weights_before_G)
    models["MLP"].load_state_dict(weights_before_MLP)
    models["MLP_cls"].load_state_dict(weights_before_MLP_cls)
