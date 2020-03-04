import cv2
import math
import numpy as np
import os
# pdb仅仅用于调试，不用管它
import pdb
import matplotlib.pyplot as plt

#旋转图像的函数
def rotate_image(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    # 仿射变换
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

data_path = '../../../data/datasets/helen/train_single_merge_dhelen'
fig, axes = plt.subplots(1, 8, figsize=(12, 12))
ax = axes.ravel()
raw_image = cv2.imread(os.path.join(data_path,'16547388_1_image.jpg'))
ax[0].imshow(raw_image)
angle_list = [60, 90, 120, 150, 210, 240, 300]
for i,angle in enumerate(angle_list):
    rotate_img = rotate_image(raw_image,angle)
    ax[i+1].imshow(rotate_img)
plt.show()
