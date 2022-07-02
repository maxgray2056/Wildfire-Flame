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

from model_list import*
from torchsummary import summary



import torch.optim as optim
from torch.utils.data import DataLoader

import datetime
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

from dataset import MyDataset


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
print(f'use device: {DEVICE}')



classes_num = 3 
BATCH_SIZE = 20
LR = 0.001
test_interval = 3
EPOCH = 50
subset_ratio = 0.1




model_name = 'FLAME_GAN'
file_name =model_name+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'_only_rgb'

path_file = './/254p Dataset//'
path_rgb = path_file+'254p RGB Images//'
path_ir = path_file+'254p Thermal Images//'
Dataset = MyDataset(path_rgb, path_ir,input_size=254)



G = Generator().to(device)
summary(G, input_size=(3,254,254))


D = Discriminator().to(device)
summary(D, input_size=(3,254,254))

    
    

#####################################################################################


#####################################################################################






# Dataset,_ = torch.utils.data.random_split(Dataset, [int(len(Dataset)*subset_ratio), len(Dataset)-int(len(Dataset)*subset_ratio)])
split_rate = 0.8
# train_set, val_set = torch.utils.data.random_split(Dataset, [int(len(Dataset)*split_rate), len(Dataset)-int(len(Dataset)*split_rate)])

train_dataloader = DataLoader(dataset=Dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_dataloader = DataLoader(dataset=val_set, batch_size=16, shuffle=False)



# optimizer = optim.Adam(model.parameters(), lr=LR,weight_decay=0.00)
optimizerG = optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# optimizerG = optim.SGD(G.parameters(), lr=LR, momentum=0.9)
# optimizerD = optim.SGD(D.parameters(), lr=LR, momentum=0.9)


loss_function_2 = nn.MSELoss()
# loss_function = nn.NLLLoss()
loss_function = nn.BCELoss()
# loss_function = CrossEntropyLoss()

# input_noise = torch.randn(BATCH, NOISE, 1, 1)
# input_noise = input_noise.to(device)
# real_label = 1.0
# fake_label = 0.0

real = 1.0
fake = 0.0



correct_on_train = [0]
correct_on_test = [0]

total_step = (int(len(Dataset)*split_rate)// BATCH_SIZE  )



'''
Training
'''
D.train()
G.train()

D_period = 5
G_period = 5
D_loss_list  = []
G_loss_list = []

max_accuracy = 0
Loss_accuracy = []
test_acc = []
loss_list = []



for epoch in range(EPOCH):
    for i, (rgb, ir, y) in enumerate(train_dataloader):

        for k in range(D_period):
            optimizerD.zero_grad()
            
            # y = y.to(DEVICE)#.unsqueeze(1)
            batch = y.shape[0]
            real_label = torch.full((batch,1,1,1), real, device=DEVICE)
    
            D_pred_RGB = D(rgb.to(DEVICE))
            D_pred_IR = D(ir.to(DEVICE))
            
            # print(real_label.shape)
            # print(D_pred_RGB.shape)
    
            loss_RGB = loss_function(D_pred_RGB, real_label)
            loss_IR = loss_function(D_pred_IR, real_label)
            loss_real = loss_RGB
            loss_real.backward()
            
            # noise = torch.randn(batch, 3, 254, 254, device=DEVICE)
            # mid = torch.cat((rgb,ir), 1)
            # mid = np.random.choice([rgb,ir], p=[0.5, 0.5])
            
            fake_image = G(ir.to(DEVICE))
            D_pred_fake = D(fake_image)
            fake_label = torch.full((batch,1,1,1), fake, device=DEVICE)
            loss_fake = loss_function(D_pred_fake, fake_label)
            loss_fake.backward()
            
            D_loss = loss_real + loss_fake
            # D_loss.backward()
            optimizerD.step()
            
            D_loss_list.append(D_loss.item())



        for j in range(G_period):
            optimizerG.zero_grad()
            # noise = torch.randn(batch, 3, 254, 254, device=DEVICE)
            # mid = torch.cat((rgb,ir), 1)
            fake_image = G(ir.to(DEVICE))
            G_pred = D(fake_image)
            # print(G_pred)
            # print(real_label)
            # G_loss = loss_function_2(fake_image, rgb.to(DEVICE))
            G_loss = loss_function(G_pred, real_label)
            G_loss_list.append(G_loss.item())
            G_loss.backward()
            optimizerG.step()
            
            
        if (i+1)%5 == 0:
            D.eval()
            G.eval()
            # noise = torch.randn(10, NOISE, 1, 1, device=device)
            # fake_images = G(noise)
            # frames = G_fake_images(fake_images)
            image_rgb  = rgb[0].permute(1,2,0).numpy()
            image_ir   = ir[0].permute(1,2,0).numpy()
            image_fake = fake_image.detach()[0].permute(1,2,0).cpu().numpy()
            
            image_frame = np.hstack((image_rgb,image_fake,image_ir))
            label_class = y[0].numpy()
            
            plt.figure(figsize=(10,10))
            plt.imshow(image_frame)
            plt.axis('off')
            plt.title('Epoch: %d niter: %d Class: %d'%(epoch,i,label_class),fontsize=16)
            plt.show()
            
            # print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (
            #     epoch, EPOCH, i, len(dataloader), D_loss.item(), G_loss.item()))
            print(f'Epoch:{epoch + 1}/{EPOCH}     Step:{BATCH_SIZE*(i+1)}|{len(train_dataloader)}   D_loss:{D_loss.item()}   G_loss:{G_loss.item()}')

            D.train()
            G.train()


            
            # image_index = random.randint(1, 53450)
            # image_test = cv2.imread(path_rgb+str(image_index)+'.jpg')
            # plt.figure(figsize=(10,10))
            # plt.imshow(image_test)
            # plt.show()
            
            # image_ref = cv2.imread(path_ir+str(image_index)+'.jpg')
            # plt.figure(figsize=(10,10))
            # plt.imshow(image_ref)
            # plt.show()
            
            # image_test = transforms.ToTensor()(image_test)
            # image_test = image_test.unsqueeze(0)
            
            # y_pre = model(image_test.to(DEVICE))
            # image_test = y_pre.detach()[0].permute(1,2,0).cpu().numpy()
            # plt.figure(figsize=(10,10))
            # plt.imshow(image_test)
            # plt.show()
            

            
            # model.train()
    torch.save(D.state_dict(), './save_weights/'+str(model_name)+'_D_'+str(epoch)+'.pth')
    torch.save(G.state_dict(), './save_weights/'+str(model_name)+'_G_'+str(epoch)+'.pth')

    # if index ==9:
    #     torch.save(net, f'saved_model/{file_name} batch={BATCH_SIZE}.pkl')
    
    # if ((index + 1) % test_interval) == 0:
        # current_accuracy = test(test_dataloader)
        # test_acc.append(current_accuracy)
        # test(train_dataloader, 'train_set')
        # print(f'current max accuracy\t test set:{max(correct_on_test)}%\t train set:{max(correct_on_train)}%')

        # if current_accuracy > max_accuracy:
        #     max_accuracy = current_accuracy
        #     torch.save(net, f'saved_model/{file_name} batch={BATCH_SIZE}.pkl')



torch.save(D.state_dict(), './save_weights/'+str(model_name)+'_D_Final.pth')
torch.save(G.state_dict(), './save_weights/'+str(model_name)+'_G_Final.pth')
















