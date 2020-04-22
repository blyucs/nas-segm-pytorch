import os
import numpy as np
import cv2
from tqdm import tqdm


def fast_histogram(a, b, na, nb):
    '''
    fast histogram calculation
    ---
    * a, b: non negative label ids, a.shape == b.shape, a in [0, ... na-1], b in [0, ..., nb-1]
    '''
    assert a.shape == b.shape
    assert np.all((a >= 0) & (a < na) & (b >= 0) & (b < nb))
    # k = (a >= 0) & (a < na) & (b >= 0) & (b < nb)
    hist = np.bincount(
        nb * a.reshape([-1]).astype(int) + b.reshape([-1]).astype(int),
        minlength=na * nb).reshape(na, nb)
    assert np.sum(hist) == a.size
    return hist


def _read_names(file_name):
    label_names = []
    for name in open(file_name, 'r'):
        name = name.strip()
        if len(name) > 0:
            label_names.append(name)
    return label_names


def _merge(*list_pairs):
    a = []
    b = []
    for al, bl in list_pairs:
        a += al
        b += bl
    return a, b

gt_label_names = pred_label_names = ['bg','face','lb','rb','le','re','nose','ulip','imouth','llip','hair',]
# gt_label_names = pred_label_names = ['bg','face','lb','rb','le','re','nose','ulip','imouth','llip',] #nohair
def cal_f1_score(gt_dir,pred_dir):

    assert gt_label_names[0] == pred_label_names[0] == 'bg'

    hists = []
    for name in tqdm(os.listdir(gt_dir)):
        if not name.endswith('.png'):
            continue

        gt_labels = cv2.imread(os.path.join(
            gt_dir, name), cv2.IMREAD_GRAYSCALE)

        pred_labels = cv2.imread(os.path.join(
            pred_dir, name), cv2.IMREAD_GRAYSCALE)
        hist = fast_histogram(gt_labels, pred_labels,
                              len(gt_label_names), len(pred_label_names))
        hists.append(hist)

    hist_sum = np.sum(np.stack(hists, axis=0), axis=0)

    eval_names = dict()
    for label_name in gt_label_names:
        gt_ind = gt_label_names.index(label_name)
        pred_ind = pred_label_names.index(label_name)
        eval_names[label_name] = ([gt_ind], [pred_ind])
    if 'le' in eval_names and 're' in eval_names:
        eval_names['eyes'] = _merge(eval_names['le'], eval_names['re'])
    if 'lb' in eval_names and 'rb' in eval_names:
        eval_names['brows'] = _merge(eval_names['lb'], eval_names['rb'])
    if 'ulip' in eval_names and 'imouth' in eval_names and 'llip' in eval_names:
        eval_names['mouth'] = _merge(
            eval_names['ulip'], eval_names['imouth'], eval_names['llip'])
    if 'eyes' in eval_names and 'brows' in eval_names and 'nose' in eval_names and 'mouth' in eval_names:
        eval_names['overall'] = _merge(
            eval_names['eyes'], eval_names['brows'], eval_names['nose'], eval_names['mouth'])
    print(eval_names)

    for eval_name, (gt_inds, pred_inds) in eval_names.items():
        A = hist_sum[gt_inds, :].sum()
        B = hist_sum[:, pred_inds].sum()
        intersected = hist_sum[gt_inds, :][:, pred_inds].sum()
        f1 = 2 * intersected / (A + B)
        print(f'f1_{eval_name}={f1}')

gt_celebA_label_names = pred_celebA_label_names = ['bg','skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
def cal_f1_score_celebA(gt_dir,pred_dir,arch_writer=None):

    assert gt_celebA_label_names[0] == pred_celebA_label_names[0] == 'bg'

    hists = []
    for name in tqdm(os.listdir(gt_dir)):
        if not name.endswith('.png'):
            continue

        gt_labels = cv2.imread(os.path.join(
            gt_dir, name), cv2.IMREAD_GRAYSCALE)

        pred_labels = cv2.imread(os.path.join(
            pred_dir, name), cv2.IMREAD_GRAYSCALE)
        hist = fast_histogram(gt_labels, pred_labels,
                              len(gt_celebA_label_names), len(pred_celebA_label_names))
        hists.append(hist)

    hist_sum = np.sum(np.stack(hists, axis=0), axis=0)

    eval_names = dict()
    for label_name in gt_celebA_label_names:
        gt_ind = gt_celebA_label_names.index(label_name)
        pred_ind = pred_celebA_label_names.index(label_name)
        eval_names[label_name] = ([gt_ind], [pred_ind])
    if 'l_eye' in eval_names and 'r_eye' in eval_names:
        eval_names['eyes'] = _merge(eval_names['l_eye'], eval_names['r_eye'])
    if 'l_brow' in eval_names and 'r_brow' in eval_names:
        eval_names['brows'] = _merge(eval_names['l_brow'], eval_names['r_brow'])
    if 'u_lip' in eval_names and 'mouth' in eval_names and 'l_lip' in eval_names:
        eval_names['mouth_a'] = _merge(
            eval_names['u_lip'], eval_names['mouth'], eval_names['l_lip'])
    if 'l_ear' in eval_names and 'r_ear' in eval_names:
        eval_names['ear'] = _merge(
            eval_names['l_ear'], eval_names['r_ear'])
    if 'eyes' in eval_names and 'brows' in eval_names and 'nose' in eval_names and 'mouth_a' in eval_names:
        eval_names['face_all'] = _merge(
            eval_names['eyes'], eval_names['brows'], eval_names['nose'], eval_names['mouth_a'])
    if 'face_all' in eval_names and 'hair' in eval_names and 'hat' in eval_names  and 'ear' in eval_names and 'ear_r' in eval_names and \
            'neck' in eval_names and 'neck_l' in eval_names and 'cloth' in eval_names and 'eye_g' in eval_names:
        eval_names['overall'] = _merge(
            eval_names['face_all'], eval_names['hair'], eval_names['hat'], eval_names['ear'], eval_names['ear_r'], eval_names['neck'] \
            ,eval_names['neck_l'], eval_names['cloth'], eval_names['eye_g'])
    print(eval_names)

    for eval_name, (gt_inds, pred_inds) in eval_names.items():
        A = hist_sum[gt_inds, :].sum()
        B = hist_sum[:, pred_inds].sum()
        intersected = hist_sum[gt_inds, :][:, pred_inds].sum()
        f1 = 2 * intersected / (A + B)
        print(f'f1_{eval_name}={f1}')
        if arch_writer != None:
            arch_writer.write(f'f1_{eval_name}=\t{f1}\n')
    if arch_writer != None:
        arch_writer.flush()
