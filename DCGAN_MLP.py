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

from models import Discriminator, Generator, MLP, MLP_cls, MLP_mean_std
from load_custom_Omniglot import customOmniglot
from utils import createAntidomain
from trainer import Trainer

matplotlib.style.use('ggplot')
plot = True

'''
La versione standard è quella che usa i due MLP
La versione noise è quella dove usa il solo rumore (per dimostrare che senza MLPO non va)
La versione mlp_mean_std usa un solo MLP e il reparametrization trick
La versione concat fa 30% classe 70% rumore
'''


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_labels', type=int, default=10, help='numbers of label in the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--image_size', type=int, default=32, help='image size')
    parser.add_argument('--nz', type=int, default=100, help='size of input noise')
    parser.add_argument('--lr', type=float, default=0.0002, help='training learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--innerepochs', type=int, default=5, help='number of meta-iterations')
    parser.add_argument('--outerstepsize0', type=float, default=0.01, help='meta learning rate')
    parser.add_argument('--niterations', type=int, default=40000, help='total training iterations')
    parser.add_argument('--sample_path', type=str, default='samples/', help='sample images folder')
    parser.add_argument('--model_path', type=str, default='models/', help='modele folder')
    parser.add_argument('--lambda_cls', type=int, default=10, help='weight of generator classification loss')
    parser.add_argument('--mode', type=str, default='standard', help='training mode - standard, mlp_mean_std, concat, noise')
    return parser.parse_args()




def create_filtered_dataloader(args,data_list):
    return DataLoader(
            data_list,
            sampler=RandomSampler(data_list, replacement=True),
            batch_size=args.batch_size,
            num_workers=4
    )


######################################################################################
def train(args):

    print("MODE: {}".format(args.mode))

    NUM_LABELS = args.num_labels

    # create samples and model folders
    if not os.path.exists(args.sample_path):
        os.mkdir(args.sample_path)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.exists("loaders_imgs/"):
        os.mkdir("loaders_imgs/")
    
    # set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE: {}".format(device))

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()#,
        #transforms.Normalize((0.1307,), (0.3081,))
        ])

    # load Omniglot
    train_dataset = datasets.Omniglot(
        root='data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.Omniglot(
        root='data',
        train=False,
        download=True,
        transform=transform
    )

    # train loader 
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=RandomSampler(test_dataset),
        batch_size=args.batch_size
    )

    # load single class loaders using a custom Omniglot loader
    loader_single_class = [create_filtered_dataloader(args, customOmniglot(root = 'data',
                        label = i, transform=transform)) 
                        for i in range(NUM_LABELS)]

    loader_single_class_test = [create_filtered_dataloader(args, customOmniglot(root = 'data',
                        label = i, train = False, transform=transform)) 
                        for i in range(NUM_LABELS)]

    # load N-1 class loaders (load each class except 1)
    loader_masked_class = [create_filtered_dataloader(args, customOmniglot(root = 'data',
                        label = i, transform=transform, mask_mode=True)) for i in range(NUM_LABELS)]

    loader_masked_class_test = [create_filtered_dataloader(args, customOmniglot(root = 'data',
                        label = i, train = False, transform=transform, mask_mode=True)) 
                        for i in range(NUM_LABELS)]
    ######################################################################################

    # for i in range(NUM_LABELS):
    #     sc_batch,_ = next(iter(loader_single_class_test[i]))
    #     masked_batch,_ = next(iter(loader_masked_class_test[i]))

    #     print("SAVING IMGS {}".format(i))
    #     save_image(sc_batch,f"loaders_imgs/sc_{i}.png")
    #     save_image(masked_batch,f"loaders_imgs/masked_{i}.png")

    # exit(0)
    # custom weights initialization 
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    model_D = Discriminator().to(device)#.cuda()
    model_D.apply(weights_init)

    model_G = Generator().to(device)#.cuda()
    model_G.apply(weights_init)

    
    model_MLP = MLP().to(device)#.cuda()
    model_MLP.apply(weights_init)

    if args.mode == 'standard' or args.mode == 'noise' or args.mode == 'concat':
        model_MLP_cls = MLP_cls().to(device)#.cuda()
        model_MLP_cls.apply(weights_init)

    if args.mode == 'mlp_mean_std':
        model_MLP_cls = MLP_mean_std().to(device)#.cuda()
        model_MLP_cls.apply(weights_init)

    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # Initialize BCELoss function
    #criterion_class = nn.BCELoss()


    # Setup Adam optimizers for all nets
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_G = optim.Adam(model_G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_MLP = optim.Adam(model_MLP.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_MLP_cls = optim.Adam(model_MLP_cls.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    loss_task_G = [[], [], [], [], [], [], [], [], [], []]
    loss_task_D = [[], [], [], [], [], [], [], [], [], []]
    loss_task_MLP = [[], [], [], [], [], [], [], [], [], []]

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

    # class that is used to train the network
    trainer = Trainer(args, models, optimizers, device, fixed_noise)

    # Reptile training loop
    for iteration in range(args.niterations):

        random = randrange(NUM_LABELS)
        
        #clone models
        weights_before_D = deepcopy(models["D"].state_dict())
        weights_before_MLP = deepcopy(models["MLP"].state_dict())
        weights_before_MLP_cls = deepcopy(models["MLP_cls"].state_dict())

        # Generate task
        single_loader = loader_single_class[random] # random from 0 to 9
        masked_loader = loader_masked_class[random]
        
        # train the GAN model with task images for N meta iterations
        err_D, err_MLP = trainer.train_GAN_on_task(single_loader, masked_loader, train_loader)
        # traing G for 1 iteration
        err_G = trainer.train_G()

        # Interpolate between current weights and trained weights from this task
        # I.e. (weights_before - weights_after) is the meta-gradient
        outerstepsize = args.outerstepsize0 * (1 - iteration / args.niterations) # linear schedule
        
        weights_after_D = models["D"].state_dict()
        models["D"].load_state_dict({name : 
            weights_before_D[name] + (weights_after_D[name] - weights_before_D[name]) * outerstepsize 
            for name in weights_before_D})

        weights_after_MLP = models["MLP"].state_dict()
        models["MLP"].load_state_dict({name : 
            weights_before_MLP[name] + (weights_after_MLP[name] - weights_before_MLP[name]) * outerstepsize 
            for name in weights_before_MLP})

        weights_after_MLP_cls = models["MLP_cls"].state_dict()
        models["MLP_cls"].load_state_dict({name : 
            weights_before_MLP_cls[name] + (weights_after_MLP_cls[name] - weights_before_MLP_cls[name]) * outerstepsize 
            for name in weights_before_MLP_cls}) 

        if iteration % 10 == 0:
            print("[{}/{}] ErrD: {:.3f} ErrG: {:.3f} ErrMLP {:.3f}".format(iteration,args.niterations,err_D,err_G, err_MLP)) 
        # Periodically plot the results on a particular task and minibatch
        #TEST
        if plot and (iteration==0 or iteration%1000 == 0 or iteration+1==args.niterations):

            for n in range(NUM_LABELS):
                weights_before_D = deepcopy(models["D"].state_dict()) # save snapshot before evaluation
                weights_before_MLP = deepcopy(models["MLP"].state_dict()) # save snapshot before evaluation
                weights_before_MLP_cls = deepcopy(models["MLP_cls"].state_dict())

                single_loader_test = loader_single_class_test[n] #selezioni il loader della classe n
                masked_loader_test = loader_masked_class_test[n]
                # update the network on a single class
                err_D, err_MLP = trainer.train_GAN_on_task(single_loader_test, masked_loader_test, train_loader)

                with torch.no_grad():
                    out_imgs_fake = trainer.generate_sample().detach().cpu()
                    
                # save the generated images
                print("SAVING IMGS {}".format(n))
                save_image(out_imgs_fake,args.sample_path + f"{iteration}_{n}.png") #aggiungi percorso: "path/iterazione_classe.png" es "pippo/20000_3.png"
                    
                # restore the model to before the update
                models["D"].load_state_dict(weights_before_D) # restore from snapshot  
                models["MLP"].load_state_dict(weights_before_MLP)
                models["MLP_cls"].load_state_dict(weights_before_MLP_cls)

        # save models
        if iteration%10000 == 0:
            torch.save(models["D"], args.model_path + "{}_D.pth".format(iteration))
            torch.save(models["G"], args.model_path + "{}_G.pth".format(iteration))
            torch.save(models["MLP"], args.model_path + "{}_MLP.pth".format(iteration))
            torch.save(models["MLP_cls"], args.model_path + "{}_MLP_cls.pth".format(iteration))


        torch.cuda.empty_cache()

if __name__ == '__main__':
    args = get_args()
    train(args)
