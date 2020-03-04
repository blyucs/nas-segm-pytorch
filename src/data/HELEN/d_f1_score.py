import os
import numpy as np
import cv2
from tqdm import tqdm
from acc import acc_eval11



if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     'gt_dir', help='the directory containing the groundtruth labels (in .png)')
    # parser.add_argument(
    #     'pred_dir', help='the directory containing the prediction labels (names should be the same with groundtruth files)')
    # args = parser.parse_args()

    # label_names_file = './label_names.txt'
    # gt_label_names = pred_label_names = _read_names(label_names_file)
    #
    # assert gt_label_names[0] == pred_label_names[0] == 'bg'

    acca = 0
    acc_bg = 0
    acc_bl = 0
    acc_br = 0
    acc_el = 0
    acc_er = 0
    acc_nose = 0
    acc_lup = 0
    acc_mi = 0
    acc_ll = 0

    acc_mouth = 0
    acc_brow = 0
    acc_eye = 0
    acc_face = 0
    num_test = 0

    # hists = []
    for name in tqdm(os.listdir("./validate_gt")):
        if not name.endswith('.png'):
            continue
        num_test+=1
        gt_labels = cv2.imread(os.path.join(
            "./validate_gt", name), cv2.IMREAD_GRAYSCALE)

        pred_labels = cv2.imread(os.path.join(
            "./validate_output", name), cv2.IMREAD_GRAYSCALE)

        accta, acct_bg, acct_bl, acct_br, acct_el, acct_er, acct_nose, acct_lup, acct_mi, acct_ll, acct_mouth, acct_brow, acct_eye, acct_face \
                = acc_eval11(eval_images=pred_labels, labels=gt_labels)
            #
        acca += accta
        acc_bg += acct_bg
        acc_bl += acct_bl
        acc_br += acct_br
        acc_el += acct_el
        acc_er += acct_er
        acc_nose += acct_nose
        acc_lup += acct_lup
        acc_mi += acct_mi
        acc_ll += acct_ll

        acc_mouth += acct_mouth
        acc_brow += acct_brow
        acc_eye += acct_eye
        acc_face += acct_face

        #
        #
        #
        #     # print (accta)
        #     # print(acct1)
        #     # print(acct2)
        #     # print(acct3)
        #     # print(acct4)
        #     # print(acct5)
        #     # print(acct6)
        #     # print(acct7)
        #     # print(acct8)
        #     # print(acct9)
        #     # print(acct_mouth)
        #     # print(acct_brow)
        #     # print(acct_eye)
        #
        #
        #
        #print('bg: ', acc_bg / num_test)

        #print('b_l: ', acc_bl / num_test)
        #print('b_r: ', acc_br / num_test)
        #print('e_l: ', acc_el / num_test)
        #print('e_r: ', acc_er / num_test)
    print("eye: ", acc_eye / num_test)
    print('brow: ', acc_brow / num_test)
    print('nose: ', acc_nose / num_test)
    print('mouth_in: ', acc_mi / num_test)
    print('lip_u: ', acc_lup / num_test)
    print('lip_l: ', acc_ll / num_test)
    print('mouth_all: ', acc_mouth / num_test)

    print('face: ', acc_face / num_test)
    print('overall: ', acca / num_test)

    print('num_test: ', num_test)
