import os
import cv2
import glob
import numpy as np
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt

# folder_base = 'CelebAMask-HQ-mask-anno'
# raw_save = 'Labels'

image_id_list = open(os.path.join('.', 'train.lst'), 'w')


for fpathe,dirs,fs in os.walk('./'):
	#cur_dir = [os.path.join(fpathe,dir) for dir in dirs if dir=='resize_img']
	for dir in dirs:
		if dir == 'resize_img':
			cur_dir = os.path.join(fpathe,dir)
			for i in os.listdir(cur_dir):
				if os.path.splitext(i)[1] == '.png':
					resize_dir = os.path.join(cur_dir,i).replace('./Supervisely_face/','')
					image_id_list.write(resize_dir+ '\t')
					mask_dir = resize_dir.replace('resize_img','resize_mask')
					image_id_list.write(mask_dir+'\t')
					image_id_list.write(mask_dir + '\n')

#for k in range(img_num):
	# folder_num = k // 2000
	# im_base = np.zeros((512, 512))
	# for idx, label in enumerate(label_list):
	# 	filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
	# 	if (os.path.exists(filename)):
	# 		#print (label, idx+1)
	# 		im = cv2.imread(filename)
	# 		im = im[:, :, 0]
	# 		im_base[im != 0] = (idx + 1)
	# filename_save = os.path.join(folder_save, str(k) + '.png')
	# print (filename_save)
	# cv2.imwrite(filename_save, im_base)

	# raw_name = os.path.join(folder_raw, str(k).rjust(5,'0') + '.png')
	# if (os.path.exists(raw_name)):
		# image=cv2.imread(raw_name)
		# image=cv2.resize(image,(512,512),interpolation=cv2.INTER_CUBIC)
		# rawname_save = os.path.join(raw_save,str(k)+'.jpg')
		# cv2.imwrite(rawname_save,image)
		# Read ids of images whose annotations have been converted from specified file
		# image_id_list = open(os.path.join('.', 'train.lst'),'w')
		# list_str = folder_raw +'/'+ str(k).rjust(5,'0')+ '.png'+'\t'+ folder_save+ '/' + \
		#            str(k).rjust(5,'0') +'.png'+'\t' + folder_save + '/' + str(k).rjust(5,'0') +'.png'
		# image_id_list.write(list_str+ '\n')

# Show image with segmentations
fig, ax = plt.subplots(figsize=[10,10])

fplot = ax.imshow(plt.imread('./SuperviselyPerson_ds1/resize_img/bodybuilder-weight-training-stress-38630.png'))
splot = ax.imshow(plt.imread('./SuperviselyPerson_ds1/resize_mask/bodybuilder-weight-training-stress-38630.png'), alpha=0.7)  # X*Y*10 imshow auto expand 0-10 to specific color
# ax.imshow(segs[i%l+2],alpha=0.7)
plt.show()