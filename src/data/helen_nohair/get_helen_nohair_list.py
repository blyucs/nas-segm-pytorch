import os
import cv2
import glob
import numpy as np
# from utils import make_folder
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

#list1
#label_list = ['skin', 'neck', 'hat', 'eye_g', 'hair', 'ear_r', 'neck_l', 'cloth', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'nose', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip']
#list2	 
#label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
# label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat',  'neck', 'cloth']

data_path = '../../../data/datasets/helen/'
folder_train = 'train_single_merge_dhelen_nohair'
folder_test = 'test_nohair'
# img_num = 30000
# folder_raw = 'CelebA-HQ-img'
# raw_save = 'CelebA-HA-img-resize'
# make_folder(folder_save)
#make_folder(raw_save)

image_train_list = os.listdir(os.path.join(data_path,folder_train))
image_val_list = os.listdir(os.path.join(data_path,folder_test))

for k,image in enumerate(image_val_list):
	if image.endswith('label.png'):
		filename = os.path.join(data_path, folder_test, image)
		if (os.path.exists(filename)):
			#print (label, idx+1)
			im = cv2.imread(filename)
			im = im[:, :, 0]
			#im_base[im != 0] = (idx + 1)
			im[im == 10] = 0 #force hair to be bg
			cv2.imwrite(filename, im)

for k,image in enumerate(image_train_list):
	if image.endswith('label.png'):
		filename = os.path.join(data_path, folder_train, image)
		if (os.path.exists(filename)):
			#print (label, idx+1)
			im = cv2.imread(filename)
			im = im[:, :, 0]
			#im_base[im != 0] = (idx + 1)
			im[im == 10] = 0 #force hair to be bg
			cv2.imwrite(filename, im)
# # Show image with segmentations
# fig, ax = plt.subplots(figsize=[10,10])
#
# fplot = ax.imshow(plt.imread(os.path.join(data_path,folder_test,'1384775111_1_label.png')))
# splot = ax.imshow(plt.imread(os.path.join(data_path,folder_test,'1384775111_1_image.jpg')), alpha=0.7) # X*Y*10 imshow auto expand 0-10 to specific color
# # ax.imshow(segs[i%l+2],alpha=0.7)
# plt.show()