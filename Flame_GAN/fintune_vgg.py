# -*- coding: utf-8 -*-
"""
Created on Mon May 23 22:11:50 2022

@author: xiwenc


fine-tine pre-trained network

"""



import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from dataset import MyDataset
from torch.utils.data import DataLoader

import datetime
import os
import random
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
print(f'use device: {DEVICE}')
## set random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
# setup_seed(20222222)  

classes_num = 3 
BATCH_SIZE = 64
LR = 0.9e-4
test_interval = 3
EPOCH =50


# model_name = 'vgg16_'
# file_name =model_name+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'only_ir'

# model = models.vgg16(pretrained=True) 
# for parameter in model.parameters():
#     parameter.required_grad = False
# model.classifier = nn.Sequential(nn.Linear(512*7*7, 4096),
#                                  nn.ReLU(inplace=True),
#                                  nn.Dropout(0.5),
#                                  nn.Linear(4096, 4096),
#                                  nn.ReLU(inplace=True),
#                                  nn.Dropout(0.5),
#                                  nn.Linear(4096, classes_num))



model_name = 'Logistic_'
file_name =model_name+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'_only_rgb'

class Logistic(nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        
        self.logic = nn.Sequential( 
            nn.Linear(254*254*3, classes_num)
        )
    
    def forward(self,x):
        x = x.view(-1, 254*254*3)
        x = self.logic(x)
        return x    
    









net = Logistic().to(DEVICE)
print(net)


path_rgb = './254p RGB Images/'
path_ir = './254p Thermal Images/'
Dataset = MyDataset(path_rgb, path_ir,input_size=224)






Dataset,_ =torch.utils.data.random_split(Dataset, [int(len(Dataset)*0.01), len(Dataset)-int(len(Dataset)*0.01)])

split_rate = 0.8
train_set, val_set = torch.utils.data.random_split(Dataset, [int(len(Dataset)*split_rate), len(Dataset)-int(len(Dataset)*split_rate)])

train_dataloader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=val_set, batch_size=16, shuffle=False)



optimizer = optim.Adam(net.parameters(), lr=LR,weight_decay=0.00)

loss_function =nn.CrossEntropyLoss(label_smoothing=0.0)
correct_on_train = [0]
correct_on_test = [0]

total_step = (int(len(Dataset)*split_rate)// BATCH_SIZE  )


def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for  (rgb, ir, y)  in dataloader:
            #print('1')
            rgb, y = rgb.to(DEVICE), y.to(DEVICE)
            # _, label_true = torch.max(y.data, dim=-1)
            y_pre= net(rgb)
            _, label_index = torch.max(y_pre.data, dim=-1)
            
            total += label_index.shape[0]
            correct += (label_index == y).sum().item()
            
        if flag == 'test_set':
            
            correct_on_test.append(round((100 * correct / total), 2))
        elif flag == 'train_set':
            #label_index = F.one_hot(label_index, num_classes=3)
            
            correct_on_train.append(round((100 * correct / total), 2))
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))
        
        return  correct / total


net.train()
max_accuracy = 0
Loss_accuracy = []
test_acc = []
loss_list = []


for index in range(EPOCH):
    net.train()
    for i, (rgb, ir, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        #print(x.shape)
        y_pre = net(rgb.to(DEVICE))
        #print(y_pre.shape)
        loss = loss_function(y_pre, y.to(DEVICE))
        Loss_accuracy.append(loss)
        print(f'Epoch:{index + 1}/{EPOCH}     Step:{i+1}|{total_step}   loss:{loss.item()}  ')
        loss_list.append(loss.item())

        loss.backward()
                                                                                          
        optimizer.step()
    
    # if index ==9:
    #     torch.save(net, f'saved_model/{file_name} batch={BATCH_SIZE}.pkl')
    
    if ((index + 1) % test_interval) == 0:
        current_accuracy = test(test_dataloader)
        test_acc.append(current_accuracy)
        test(train_dataloader, 'train_set')
        print(f'current max accuracy\t test set:{max(correct_on_test)}%\t train set:{max(correct_on_train)}%')

        if current_accuracy > max_accuracy:
            max_accuracy = current_accuracy
            torch.save(net, f'saved_model/{file_name} batch={BATCH_SIZE}.pkl')

   

os.rename(f'saved_model/{file_name} batch={BATCH_SIZE}.pkl',
          f'saved_model/{file_name} {max_accuracy} batch={BATCH_SIZE}.pkl')


torch.save(net, f'saved_model/{file_name} final batch={BATCH_SIZE}.pkl')



