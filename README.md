# Wildfire & Flame Project
Wildfire flame detection based on deep learning & image processing

## Table of Contents
- [Background Information](#background)
- [Flame Detection](#flame-detection)
- [Flame Images Generating](#flame-images-generating)
- [Install](#install)
- [Packages](#packages)
- [Contributing](#contributing)


## Background
In this project, we propose a drone-based wildfire monitoring system for remote and hard-to-reach areas. This system utilizes autonomous unmanned aerial vehicles (UAVs) with the main advantage of providing on-demand monitoring service faster than the current approaches of using satellite images, manned aircraft and remotely controlled drones.

<img src="https://github.com/bot0231019/Wildfire-Flame/blob/main/flame.jpg" width="500px">


Avalible datasets for now:
<a href="https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs">THE FLAME DATASET: AERIAL IMAGERY PILE BURN DETECTION USING DRONES (UAVS)</a>


More information, please reference our previous paper:
<a href="https://ieeexplore.ieee.org/abstract/document/8845309">Wildfire Monitoring in Remote Areas using Autonomous Unmanned Aerial Vehicles</a>


## Flame Detection
This module only uses image processing method to detect flame zone on IR pictures, then draw bounding boxes on related RGB images.

For codes, please go to: [Flame_Detection](Flame_detection)

<img src="https://github.com/bot0231019/Wildfire-Flame/blob/main/Flame_detection/plot/Figure%202022-06-30%20001211.png" width="500px">




## Flame Images Generating
This module uses GAN-based deep learning method to generate RGB flame images from given IR image. For example, input an arbitrary IR image of wildfire, the model will generate a virtual RGB image to describe the fire situation.

This part is only on experimental stage, but it's good to research the relationship between the wildfire and its smoke.

For codes, please go to: [Flame_GAN](Flame_GAN)

<img src="https://github.com/bot0231019/Wildfire-Flame/blob/main/Flame_GAN/plot/Figure%202022-06-16%20233050.png" width="500px">






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
<a href="bryceh@g.clemson.edu">bryceh@clemson.edu</a>

### Please cite our work if you think this project helps your research.
