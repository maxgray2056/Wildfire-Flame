# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:27:30 2022

@author: hao9
"""

import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from glob import glob
import shutil
import cv2
import random
import copy

kernel_size = 7
fire_thred = 0.9
zone_overlap_thred = 0.9
flame_thred = 0.1
num_areas = 100




path_image_RGB = './/254p Dataset//254p RGB Images//'
path_image_IR  = './/254p Dataset//254p Thermal Images//'

image_index = random.randint(1,53450)
ID = image_index
if (1<= ID and ID <=13700):
    y = 0
elif   (13701	<= ID and ID <=14699) \
    or (16226	<= ID and ID <=19802) \
    or (19900	<= ID and ID <=27183) \
    or (27515	<= ID and ID <=31294) \
    or (31510	<= ID and ID <=33597) \
    or (33930	<= ID and ID <=36550) \
    or (38031	<= ID and ID <=38153) \
    or (43084	<= ID and ID <=45279) \
    or (51207	<= ID and ID <=52286):
        
    y = 1
else:
    y = 2
    
print(image_index, y)


# image_index = 15024

plt.figure(figsize=(10,10),dpi=300)
plt.suptitle('ID = '+str(image_index)+' Class = '+str(y), fontsize=20)


image_RGB = cv2.imread(path_image_RGB+str(image_index)+'.jpg')
image = cv2.imread(path_image_IR+str(image_index)+'.jpg')
# ax1 = plt.subplot(2,2,1)
# ax1.imshow(cv2.cvtColor(image_RGB, cv2.COLOR_BGR2RGB))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
w,h = gray.shape
# ax2 = plt.subplot(2,2,2)
# ax2.imshow(gray)
# ax1.colorbar()

print(np.mean(gray))
print(np.median(gray))
print(np.max(gray))

thred = np.mean(gray) + np.median(gray)

# np.sort(gray.reshape(-1))
gray_sort = np.sort(gray.reshape(-1))


fire_line = gray_sort[int(len(gray_sort)*fire_thred)]

# X2 = gray_sort
# F2 = np.array(range(len(X2)))/float(len(X2))
# plt.plot(X2, F2)

# plt.scatter(np.arange(len(gray.reshape(-1))),np.sort(gray.reshape(-1)))

seg = copy.deepcopy(gray)

seg[seg < fire_line] = 0
# plt.imshow(seg)

seg_blur = cv2.blur(seg, (kernel_size,kernel_size))



# seg_fire = copy.deepcopy(seg_blur)
# seg_fire[seg_fire < fire_line] = 0
# plt.imshow(seg_fire)




if y == 0:
    boxes = []
else:
    MSER = cv2.MSER_create(min_area = 300)
    reg, boxes = MSER.detectRegions(seg_blur)

def Nlargest(data, N):
    if type(data) != list:
        data = data.tolist()
    max_number = []
    max_index = []
    for i in range(N):
        number = max(data)
        index = data.index(number)
        data[index] = 0
        max_number.append(number)
        max_index.append(index)
    return max_number, max_index


def Flame_ratio(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]

    zone = seg_blur[y1:y2,x1:x2]
    w,h = zone.shape
    # plt.imshow(zone)
    
    flame_ratio = np.sum(zone[zone>0])/((w*h+1)*255)
    # flame_ratio = flame_ratio*np.sum(zone[zone>0])
    return flame_ratio



def NMS(boxes, overlapThresh, num_areas, flame_thred):
    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    else:
        boxes[:, 2] = boxes[:, 2]+boxes[:, 0]
        boxes[:, 3] = boxes[:, 3]+boxes[:, 1]
                
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts

    areas_order = np.argsort(areas)[::-1]
    # areas_list = areas[areas_order]
    boxes = boxes[areas_order]

    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    
    for i,box in enumerate(boxes): #print(i,box)
        # Create temporary indices  
        temp_indices = indices[indices!=i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]
            
    #return only the boxes at the remaining indices
    # areas = areas[indices].astype(int)
    boxes = boxes[indices].astype(int)
        
    flame_list = []
    for i in range(len(boxes)):
        flame_ratio = Flame_ratio(boxes[i])
        flame_list.append(flame_ratio)
    flame_list = np.array(flame_list)
    flame_order = np.argsort(flame_list)[::-1]
    flame_list = flame_list[flame_order]
    boxes = boxes[flame_order]

    print(flame_list)
    print(boxes)
    boxes = np.delete(boxes, np.where(flame_list < flame_thred), 0)


    boxes[:, 2] = boxes[:, 2]-boxes[:, 0]
    boxes[:, 3] = boxes[:, 3]-boxes[:, 1]
    
    # max_flame = Nlargest(flame_list, num_areas)[1]
    # boxes = boxes[max_flame]
    
    boxes = boxes[0:num_areas]
    return boxes




# flame_list = []
# for i in range(len(boxes)):
#     flame_ratio = Flame_ratio(boxes[i])
#     flame_list.append(flame_ratio)
# flame_list = np.array(flame_list)

# flame_order = np.argsort(flame_list)[::-1]
# flame_list = flame_list[flame_order]

# boxes = boxes[flame_order]





boxes_new = NMS(boxes, zone_overlap_thred, num_areas, flame_thred)

print(len(boxes))
print(len(boxes_new))


for i in boxes_new:
    x,y,w,h = i
    cv2.rectangle(image_RGB, (x,y),(x+w,y+h), (0,0,255), 2)
    cv2.rectangle(seg_blur,  (x,y),(x+w,y+h), (255,255,255), 2)

# plt.imshow(image_RGB)
# plt.imshow(seg_blur)


ax1 = plt.subplot(2,2,1)
ax1.imshow(cv2.cvtColor(image_RGB, cv2.COLOR_BGR2RGB))
ax1.set_title('RGB')


ax2 = plt.subplot(2,2,2)
ax2.imshow(gray)
ax2.set_title('IR')


ax3 = plt.subplot(2,2,3)
ax3.scatter(np.arange(len(gray_sort)),
            gray_sort,linewidth=5, 
            c=gray_sort, cmap='viridis')
ax3.axhline(y=fire_line, linewidth=5, color='r')
ax3.set_title('Fire Line')


ax4 = plt.subplot(2,2,4)
ax4.imshow(seg_blur)
ax4.set_title('Fire Detection')




plt.show()






