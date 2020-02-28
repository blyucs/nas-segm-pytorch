import os
import cv2
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
import shutil
import matplotlib.pyplot as plt

def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))
    else:
        shutil.rmtree(path)
        os.makedirs(path)
# no hair
label_list = ['lbl01','lbl02','lbl03','lbl04','lbl05','lbl06','lbl07','lbl08','lbl09'] # face left and right define does not match in celebA and helen
# gt_label_names = pred_label_names = ['bg','face','lb','rb','le','re','nose','ulip','imouth','llip','hair',]
folder_base = '../../../data/datasets/dhelen'
folder_anno = 'labels'
folder_raw = 'images'
folder_save = 'masks'
img_num = 30000
# folder_raw = 'CelebA-HQ-img'
raw_save = 'CelebA-HA-img-resize'
make_folder(os.path.join(folder_base,folder_save))

# make_folder(raw_save)

image_list = os.listdir(os.path.join(folder_base,folder_raw))

for k,image in enumerate(image_list):
    size = cv2.imread(os.path.join(folder_base,folder_raw,image)).shape
    im_base = np.zeros(size)
    cur_label_folder = image.split('.')[0]
    # label_list = os.listdir(os.path.join(folder_base,folder_anno,cur_label_folder))
    for idx, label in enumerate(label_list):
        label_name = os.path.join(folder_base,folder_anno,cur_label_folder, cur_label_folder+'_'+label+'.png')
        if (os.path.exists(label_name)):
            im = cv2.imread(label_name)
            im = im[:, :, 0]
            # im_base[im != 0] = (idx+1)
            im_base[im > 200] = (idx + 1)
    filename_save = os.path.join(folder_base,folder_save, cur_label_folder + '.png')
    print (filename_save)
    cv2.imwrite(filename_save, im_base)

    # raw_name = os.path.join(folder_base,raw_save, str(k) + '.jpg')
    # if (os.path.exists(raw_name)):
    # 	# image=cv2.imread(raw_name)
    # 	# image=cv2.resize(image,(512,512),interpolation=cv2.INTER_CUBIC)
    # 	# rawname_save = os.path.join(raw_save,str(k)+'.jpg')
    # 	# cv2.imwrite(rawname_save,image)
    # 	# Read ids of images whose annotations have been converted from specified file
    # 	# image_id_list = open(os.path.join('.', 'train.lst'),'w')
    # 	list_str = raw_save +'/'+ str(k)+ '.jpg'+'\t'+ folder_save+ '/' + str(k) +'.png'+'\t' + folder_save + '/' + str(k) +'.png'
    # 	image_id_list.write(list_str+ '\n')

# # Show image with segmentations
# fig, ax = plt.subplots(figsize=[10,10])
#
# fplot = ax.imshow(plt.imread(raw_save+'/'+'1.jpg'))
# splot = ax.imshow(plt.imread(folder_save+'/'+'1.png'), alpha=0.7)  # X*Y*10 imshow auto expand 0-10 to specific color
# # ax.imshow(segs[i%l+2],alpha=0.7)
# plt.show()