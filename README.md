# Wildfire & Flame Project
Wildfire flame detection based on deep learning & image processing

## Table of Contents
- [Flame Detection](#Flame_detection)
- [Flame Image Generating](#Flame_GAN)
- [Background Information](#Background)
- [Install](#install)
- [Packages](#packages)
- [Contributing](#contributing)


## Background
TBD


## Flame Detection
#Flame_detection/plot/Figure 2022-06-30 001211.png
![image](https://github.com/bot0231019/Wildfire-Flame/blob/main/Flame_detection/plot/Figure%202022-06-30%20001211.png)

## Install
This project uses Python 3.8 based on Spyder

This project is based on Pytorch 1.7.0 & cudnn 1.10
Official website: <a href="https://pytorch.org/get-started/previous-versions/">Previous PyTorch Versions</a>
```
pip install -f https://download.pytorch.org/whl/cu110/torch_stable.html torch==1.7.0+cu110 torchvision==0.8.0 --user
```



## Packages
This list gives all recommended packages (may not necessary)

### For GAN:
```sh
import datetime
import os
import random
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

import torch  
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.nn import ModuleList
from torch.nn.modules.loss import _WeightedLoss

from model_list import*
from torchsummary import summary
from torch.utils.data import DataLoader
```


### For Flame detection:
```sh
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import shutil
import cv2
import random
import copy
from glob import glob
```


## Contributing
This project is contributed by: 
<a href="hao9@g.clemson.edu">hao9@clemson.edu</a>
<a href="xiwenc@g.clemson.edu">xiwenc@clemson.edu</a>

