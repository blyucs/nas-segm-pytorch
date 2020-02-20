import os
import cv2
import glob
import numpy as np
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt


train_image_id_list = open(os.path.join('.', 'train.lst'), 'w')
val_image_id_list = open(os.path.join('.','val.lst'), 'w')
for fpathe,dirs,files in os.walk('./train'):
	for file in files:
		if file.endswith('image.jpg'):
			train_image_id_list.write('train/'+file+'\t')
			train_image_id_list.write('train/'+file.replace('image.jpg','label.png')+'\t')
			train_image_id_list.write('train/' + file.replace('image.jpg', 'label.png') + '\n')

for fpathe,dirs,files in os.walk('./test_resize'):
	for file in files:
		if file.endswith('image.jpg'):
			val_image_id_list.write('train/'+file+'\t')
			val_image_id_list.write('train/'+file.replace('image.jpg','label.png')+'\t')
			val_image_id_list.write('train/' + file.replace('image.jpg', 'label.png') + '\n')

# # Show image with segmentations
# # fig, ax = plt.subplots(figsize=[10,10])
# #
# # fplot = ax.imshow(plt.imread(folder_raw+'/'+'00002.png'))
# # splot = ax.imshow(plt.imread(folder_save+'/'+'00002.png'), alpha=0.7)  # X*Y*10 imshow auto expand 0-10 to specific color
# # ax.imshow(segs[i%l+2],alpha=0.7)
# plt.show()