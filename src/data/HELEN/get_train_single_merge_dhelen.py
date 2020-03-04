import os
import cv2
import glob
import numpy as np
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt
import fnmatch
import shutil
data_path = '../../../data/datasets/helen/'

train_image_id_list = open(os.path.join(data_path, 'train_single_merge_dhelen.lst'), 'w')
#remove the duplicate files
# for fpathe,dirs,files in os.walk(os.path.join(data_path,'train_single_merge_dhelen')):
# 	for file in files:
# 		list_dir = list(sorted(os.listdir(os.path.join(data_path, 'train_single_merge_dhelen'))))
# 		match_cur_list = fnmatch.filter(list_dir,file.split('_')[0]+'*')
# 		if(len(match_cur_list) > 3):
# 			[os.remove(os.path.join(data_path,'train_single_merge_dhelen',i)) for i in match_cur_list]

for fpathe, dirs, files in os.walk(os.path.join(data_path, 'train_single_merge_dhelen')):
	for file in files:
		if file.endswith('image.jpg'):
			file_name = file.split()
			train_image_id_list.write('train_single_merge_dhelen/'+file+'\t')
			train_image_id_list.write('train_single_merge_dhelen/'+file.replace('image.jpg','label.png')+'\t')
			train_image_id_list.write('train_single_merge_dhelen/' + file.replace('image.jpg', 'label.png') + '\n')

# # Show image with segmentations
# # fig, ax = plt.subplots(figsize=[10,10])
# #
# # fplot = ax.imshow(plt.imread(folder_raw+'/'+'00002.png'))
# # splot = ax.imshow(plt.imread(folder_save+'/'+'00002.png'), alpha=0.7)  # X*Y*10 imshow auto expand 0-10 to specific color
# # ax.imshow(segs[i%l+2],alpha=0.7)
# plt.show()