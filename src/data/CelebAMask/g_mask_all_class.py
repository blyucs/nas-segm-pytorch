import os
import cv2
import glob
import numpy as np
from utils import make_folder
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt
#list1
# label_list = ['skin', 'neck', 'hat', 'eye_g', 'hair', 'ear_r', 'neck_l', 'cloth', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'nose', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip']
#list2	 
label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
#label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat',  'neck', 'cloth']
folder_base = '../../../data/datasets/celebA'
folder_anno = 'CelebAMask-HQ-mask-anno'
folder_save = 'CelebAMask-HQ-mask-all-class'
img_num = 30000
# folder_raw = os.path.join(folder_base,'CelebA-HQ-img')
raw_img_dir = 'CelebA-HA-img-resize'
make_folder(folder_save)
# make_folder(raw_save)

image_id_list = open(os.path.join(folder_base, 'train_all_class.lst'), 'w')
test_id_list = open(os.path.join(folder_base, 'test_all_class.lst'), 'w')

for k in range(img_num):
	# folder_num = k // 2000
	# im_base = np.zeros((512, 512))
	# for idx, label in enumerate(label_list):
	# 	filename = os.path.join(folder_base,folder_anno, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
	# 	if (os.path.exists(filename)):
	# 		#print (label, idx+1)
	# 		im = cv2.imread(filename)
	# 		im = im[:, :, 0]
	# 		im_base[im != 0] = (idx + 1)
	# filename_save = os.path.join(folder_save, str(k) + '.png')
	# print (filename_save)
	# cv2.imwrite(filename_save, im_base)

	raw_name = os.path.join(folder_base, raw_img_dir, str(k) + '.jpg')
	if (os.path.exists(raw_name)):
		# image=cv2.imread(raw_name)
		# image=cv2.resize(image,(512,512),interpolation=cv2.INTER_CUBIC)
		# rawname_save = os.path.join(raw_save,str(k)+'.jpg')
		# cv2.imwrite(rawname_save,image)
		# Read ids of images whose annotations have been converted from specified file
		# image_id_list = open(os.path.join('.', 'train.lst'),'w')
		list_str = raw_img_dir +'/'+ str(k)+ '.jpg'+'\t'+ folder_save+ '/' + str(k) +'.png'+'\t' + folder_save + '/' + str(k) +'.png'
		if k > 28999 and k < 29100:
			test_id_list.write(list_str + '\n')
		else:
			image_id_list.write(list_str + '\n')

# Show image with segmentations
fig, ax = plt.subplots(figsize=[10,10])

fplot = ax.imshow(plt.imread(os.path.join(folder_base,raw_img_dir,'1.jpg')))
splot = ax.imshow(plt.imread(os.path.join(folder_base, folder_save, '1.png')), alpha=0.7)  # X*Y*10 imshow auto expand 0-10 to specific color
# ax.imshow(segs[i%l+2],alpha=0.7)
plt.show()