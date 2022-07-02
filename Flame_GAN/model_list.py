# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:36:22 2022

@author: hao9
"""



import torch  
import torch.nn as nn
import numpy as np
# import PIL.Image as Image
import torchvision.transforms as transforms
# from torch import optim
# from torch.autograd import Variable
import torch.nn.functional as F

from torch.nn import Module
#from module.feedForward import FeedForward
#from module.multiHeadAttention import MultiHeadAttention
from torch.nn import CrossEntropyLoss
from torch.nn import ModuleList
import math
from torch.nn.modules.loss import _WeightedLoss
import torchvision.models as models

from torchsummary import summary


device = torch.device("cuda")


# the architecture of the gan
class Generator_SR(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_init = nn.Sequential( 
            nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2) 
            )
        
        self.conv_1 = nn.Sequential(   
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2) 
            )
        
        self.conv_2 = nn.Sequential(   
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2) 
            )
        
        self.conv_nonlinear = nn.Sequential(   
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(True)
            )
        
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(16, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 2, stride=2, padding=0, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            )
        
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            )
        
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            )
        
        # self.deconv_4 = nn.Sequential(
        #     nn.Conv2d(32, 16, 3, stride=1, padding=2), nn.BatchNorm2d(16), nn.ReLU(True),
        #     nn.Conv2d(16, 16, 3, stride=1, padding=2), nn.BatchNorm2d(16), nn.ReLU(True),
        #     #nn.Upsample(scale_factor=2, mode='bilinear')
        #     nn.ConvTranspose2d(16, 16, 2, stride=2, padding=0, output_padding=0) 
        #     )
        
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 3, 3, stride=1, padding=1), nn.Tanh()
            )
        
    def forward(self,x):
        x = self.conv_init(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_nonlinear(x)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        # x = self.deconv_5(x)
        
        return x
        
    


# G = Generator_SR().to(device)
# summary(G, input_size=(3,254,254))




class Discriminator_SR(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_init = nn.Sequential( 
            nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2) )
        
        self.conv_1 = nn.Sequential(   
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2) )
        
        self.conv_2 = nn.Sequential(   
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2) )
        
        self.conv_nonlinear = nn.Sequential(   
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16)
            #, nn.Sigmoid()
        )
        
        self.fc = nn.Sequential(            
            nn.Linear(16, 1) )
        
        
             
    def forward(self,x):
        x = self.conv_init(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_nonlinear(x)
        x = F.adaptive_avg_pool2d(x, (1,1)) #global pooling
        # print(x.shape)
        x =  x.view(-1,16)
        # print(x.shape)
        x = self.fc(x)

        return x
    

# D = Discriminator_SR().to(device)
# summary(D, input_size=(3,254,254))

    
class AE_3(nn.Module):
    def __init__(self):
        super(AE_3, self).__init__()
        
        self.conv_init = nn.Sequential( 
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        
        self.conv_2 = nn.Sequential(   
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        
        self.conv_3 = nn.Sequential(   
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        
        self.conv_nonlinear = nn.Sequential(   
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 16, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        
        
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(16, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 128, 2, stride=2, padding=0, output_padding=1)
        )
        
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, output_padding=1)
        )
        
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0, output_padding=0)
        )
        
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
        )
        
    
    def forward(self,x):
        x = self.conv_init(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_nonlinear(x)
        
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        return x    
    
# G_AE = AE_3().to(device)
    

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),

            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.Conv2d(16, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),

            # nn.Conv2d(16, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Conv2d(32, 64, 5, stride=1, padding=1), nn.ReLU(True),
            nn.Conv2d(64,  3, 3, stride=1, padding=1), nn.ReLU(True),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

    def initialize_weights(self, w_mean=0, w_std=0.02, b_mean=1, b_std=0.02):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, w_mean, w_std)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, b_mean, b_std)
                nn.init.constant_(m.bias.data, 0)

# G = Generator().to(device)
# summary(G, input_size=(3,254,254))



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

    def initialize_weights(self, w_mean=0, w_std=0.02, b_mean=1, b_std=0.02):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, w_mean, w_std)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, b_mean, b_std)
                nn.init.constant_(m.bias.data, 0)

# D = Discriminator().to(device)
# summary(D, input_size=(3,254,254))


    
    

















