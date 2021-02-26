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



class Trainer(object):
    def __init__(self, args, models, optimizers, device, fixed_noise):
        
        self.device = device
        self.fixed_noise = fixed_noise

        self.model_D = models["D"]
        self.model_G = models["G"]
        self.model_MLP = models["MLP"]
        self.model_MLP_cls = models["MLP_cls"]

        self.optimizer_D = optimizers["D"]
        self.optimizer_G = optimizers["G"]
        self.optimizer_MLP = optimizers["MLP"]
        self.optimizer_MLP_cls = optimizers["MLP_cls"]

        self.criterion = nn.BCELoss()
        
        # Establish convention for real and fake labels during training
        #self.real_label = 1.
        #self.fake_label = 0.

        self.batch_size = args.batch_size
        self.nz = args.nz
        self.innerepochs = args.innerepochs
        self.lambda_cls = args.lambda_cls
        self.mode = args.mode

        self.real_label = torch.full((self.batch_size,), 1., dtype=torch.float, device=self.device)
        self.fake_label = torch.full((self.batch_size,), 0., dtype=torch.float, device=self.device)
        
        self.out_z = 0

    def reset_grad(self):
        self.model_D.zero_grad()
        self.model_G.zero_grad()
        self.model_MLP.zero_grad()
        self.model_MLP_cls.zero_grad()

    def KLD_loss(self, log_var, mu):
        return -0.5 * torch.sum(1 + torch.log(log_var ** 2) - mu ** 2 - log_var ** 2)
    
    def std_loss(self, std, alpha = 0.3):
        return torch.mean(torch.pow(torch.clamp(alpha - std**2, min=0.0), 2))
    def std_loss_noise(self, noise, alpha = 0.3):
        return torch.mean(torch.pow(torch.clamp(alpha - torch.std(noise), min=0.0), 2))

    def MLP_pass(self, noise, feat):

        if self.mode == 'standard':
            out_mlp = self.model_MLP(noise.view(noise.size(0),-1))
            out_mlp = out_mlp.reshape(out_mlp.size(0),self.nz,1,1)

            out_mlp_cls = self.model_MLP_cls(feat.detach())
            out_mlp_cls = out_mlp_cls.reshape(out_mlp_cls.size(0),out_mlp_cls.size(1),1,1)

            return torch.cat((out_mlp, out_mlp_cls), 1)
        
        elif self.mode == 'mlp_mean_std':
            mean, std = self.model_MLP_cls(feat.detach())
            #out_mlp = torch.exp(0.5 * std) * noise.view(noise.size(0),-1) + mean
            out_mlp = std * noise.view(noise.size(0),-1) + mean

            #self.std_err = self.std_loss_noise(out_mlp)

            self.std = std
            self.mean = mean

            return out_mlp.reshape(self.batch_size,self.nz,1,1)
        elif self.mode == 'concat':
            out_mlp_cls = self.model_MLP_cls(feat.detach())
            out_mlp_cls = out_mlp_cls.reshape(out_mlp_cls.size(0),out_mlp_cls.size(1),1,1)
            return torch.cat((noise, out_mlp_cls), 1)

        elif self.mode == 'noise':
            return noise

    def update_MLP(self, err_mlp):

        #if self.mode == 'mlp_mean_std':
        #    err_mlp = err_mlp + self.std_err 

        err_mlp.backward()

        if self.mode == 'standard':
            self.optimizer_MLP.step()
            self.optimizer_MLP_cls.step()
        elif self.mode == 'mlp_mean_std' or self.mode == 'concat':
            self.optimizer_MLP_cls.step()
        elif self.mode == 'noise':
            return

    def train_G(self):
        
            # (2) Update G network: maximize log(D(G(z)))

        self.reset_grad()
        #self.label.fill_(self.real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        
        noise = torch.randn(self.batch_size, self.nz, 1, 1).cuda()
        out_z = self.MLP_pass(noise, self.feat)
        
        fake = self.model_G(out_z.detach())

        output, out_cls, _ = self. model_D(fake)
        output = output.view(-1)
        out_cls = out_cls.view(-1)
        # Calculate G's loss based on this output
        errG = self.criterion(output, self.real_label)
        err_out_cls = self.criterion(out_cls, self.real_label)
        #Classify out_cls as real
        errG = errG + self.lambda_cls*err_out_cls
        
        
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        self.optimizer_G.step()

        return errG


    def train_GAN_on_task(self, single_loader, masked_loader, train_loader):
        for i in range(self.innerepochs):
            data = next(iter(single_loader))
            masked_data = next(iter(masked_loader))
            data_full = next(iter(train_loader))
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            self.reset_grad()
            # Format batch
            real_cpu = data[0].to(self.device)
            real_full = data_full[0].to(self.device)
            real_masked = masked_data[0].to(self.device)

            if real_cpu.size(0) != self.batch_size:
                print("wrong batch size")
                continue

            # Forward pass real batch through D
            output, out_cls, feat = self.model_D(real_cpu)
            
            output = output.view(-1)
            out_cls = out_cls.view(-1)
            # Calculate loss on all-real batch
            errD_real = self.criterion(output, self.real_label)
            err_out_cls = self.criterion(out_cls,self.real_label)
            #Classify out_cls as real
            errD_real = errD_real + err_out_cls
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)
            # Generate fake image batch with G
            out_z = self.MLP_pass(noise, feat)
            fake = self.model_G(out_z)
            # Classify all fake batch with D
            output, _, _ = self.model_D(fake.detach())
            output = output.view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = self.criterion(output, self.fake_label)
            # classify out_cls as wrong domain
            _, out_cls, masked_feat = self.model_D(real_masked)
            out_cls=out_cls.view(-1)
            # calculate criterion
            err_out_cls = self.criterion(out_cls, self.fake_label)
            # Add the gradients from the all-real and all-fake domain
            errD_fake = errD_fake + err_out_cls
            #backpropagate
            errD_fake.backward()
            # Update D
            self.optimizer_D.step()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake


            self.reset_grad()
            out_real, out_dom, _ = self.model_D(fake)

            #self.label.fill_(self.real_label)
            out_real=out_real.view(-1)
            out_dom=out_dom.view(-1)
            errD_real = self.criterion(out_real, self.real_label) #il label reale
            err_out_cls = self.criterion(out_dom, self.real_label)
            

            err_mlp = errD_real + err_out_cls
            

            self.update_MLP(err_mlp)
            
        # Output training stats
            self.feat = feat
            self.fake = fake
            self.real_cpu = real_cpu
        return errD, err_mlp


    def generate_sample(self):                
        self.out_z = self.MLP_pass(self.fixed_noise, self.feat) 
        out_G = self.model_G(self.out_z.detach())  
        # return real images concatenadet with generated images    
        return torch.cat((self.real_cpu, out_G), dim = 0)
    
    def get_z(self):
        return self.out_z

    def get_mean_std(self):
        return self.mean, self.std