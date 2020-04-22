"""Main file for test.

KD from RefineNet-Light-Weight-152 (args.do_kd => keep in memory):
  Task0 - pre-computed
  Task1 - on-the-fly

Polyak Averaging (args.do_polyak):
  Task0 - only decoder
  Task1 - encoder + decoder

Search:
  Task0 - task0_epochs - validate every epoch
  Task1 - task1_epochs - validate every epoch

"""

# general libs
import argparse
import logging
import os
import random
import time
import numpy as np
import sys
# pytorch libs
import torch
import torch.nn as nn
import datetime
# custom libs
from data.loaders import create_loaders
from engine.inference import validate
from engine.trainer import populate_task0, train_task0, train_segmenter
from helpers.utils import apply_polyak, compute_params, init_polyak, load_ckpt, \
                          Saver, TaskPerformer, seg_Saver
from nn.encoders import create_encoder
from nn.micro_decoders import MicroDecoder as Decoder
from rl.agent import create_agent, train_agent
from utils.default_args import *
from utils.solvers import create_optimisers
from PIL import  Image
import cv2
import matplotlib.pyplot as plt
import  shutil
from utils.helpers import prepare_img
from utils.f1_score import *
from thop import profile
from thop import clever_format
os.environ["CUDA_VISIBLE_DEVICES"]="3"
logging.basicConfig(level=logging.INFO)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cmap = np.load('./utils/cmap.npy')
SEGMENTER_CKPT_PATH = \
    {
        # 'celebA':'./ckpt/20200111T1841/segmenter_checkpoint.pth.tar',
        # 'celebA':'./ckpt/_train_celebA_20200304T2257/segmenter_checkpoint_0.22.pth.tar',  # 19 classes
        # 'celebA':'./ckpt/_train_celebA_20200305T1410/segmenter_checkpoint_0.21.pth.tar',  # 19 classes
        # 'celebA':'./ckpt/_train_celebA_20200305T1556/segmenter_checkpoint_0.31.pth.tar',  # 19 classes
        # 'celebA':'./ckpt/_train_celebA_20200305T1653/segmenter_checkpoint_0.24.pth.tar',  # 19 classes
        # 'celebA':'./ckpt/_train_celebA_20200305T1751/segmenter_checkpoint_0.25.pth.tar',  # 19 classes 0.943 best?
        # 'celebA':'./ckpt/_train_celebA_20200305T2113/segmenter_checkpoint_0.22.pth.tar',  # 19 classes 0.945 best?
        'celebA':'./ckpt/_train_celebA_20200412T1303/segmenter_checkpoint_0.22.pth.tar',  # 19 classes
        # 'celebA':'./ckpt/_train_celebA_20200415T1358/segmenter_checkpoint_0.29.pth.tar',  # 19 classes
        # 'celebA':'./ckpt/_train_celebA_20200415T1314/segmenter_checkpoint_0.43.pth.tar',  # 19 classes
        #'EG1800':'./ckpt/train20200117T1958/segmenter_checkpoint.pth.tar'
        #'EG1800':'./ckpt/train20200118T1128/segmenter_checkpoint.pth.tar' , # 00079,00094,good, the best model currently
        #'EG1800': './ckpt/_train_EG1800_20200217T1059/segmenter_checkpoint.pth.tar',
        #'EG1800': './ckpt/_train_EG1800_20200217T1405/segmenter_checkpoint.pth.tar',
        # 'EG1800':'./ckpt/train20200118T1224/segmenter_checkpoint.pth.tar'
        #'EG1800':'./ckpt/train20200118T1239/segmenter_checkpoint.pth.tar'
        #'EG1800': './ckpt/_train_celebA-binary_20200118T1715/segmenter_checkpoint.pth.tar',# 00079,00094,good, the best model currently
        # 'celebA-binary': './ckpt/_train_celebA-binary_20200118T1715/segmenter_checkpoint.pth.tar',# 00079,00094,good, the best model currently
        #'EG1800': './ckpt/_train_EG1800_20200217T1922/segmenter_checkpoint.pth.tar',
        'celebA-binary': './ckpt/_train_EG1800_20200217T1922/segmenter_checkpoint.pth.tar',
        #'EG1800': './ckpt/_train_EG1800_20200217T2216/segmenter_checkpoint.pth.tar',
        # 'EG1800': './ckpt/_train_EG1800_20200218T1319/segmenter_checkpoint.pth.tar', #0.966
        # 'EG1800': './ckpt/_train_EG1800_20200218T1503/segmenter_checkpoint.pth.tar',
        #'EG1800': './ckpt/_train_EG1800_20200218T1606/segmenter_checkpoint.pth.tar',
        # 'EG1800': './ckpt/_train_EG1800_20200218T1842/segmenter_checkpoint.pth.tar',
        'EG1800': './ckpt/_train_EG1800_20200218T2034/segmenter_checkpoint.pth.tar',  #0.967
        # 'EG1800': './ckpt/_train_EG1800_20200218T2158/segmenter_checkpoint.pth.tar',  #0.873
        # 'helen': './ckpt/_train_helen_20200223T1724/segmenter_checkpoint.pth.tar',  #0.81
        # 'helen': './ckpt/_train_helen_20200224T1611/segmenter_checkpoint.pth.tar',  # 0.81
        # 'helen': './ckpt/_train_helen_20200225T1319/segmenter_checkpoint.pth.tar',  # no pre-trained mobilenetV2 poor performance
        # 'helen': './ckpt/_train_celebA-face_20200225T1518/segmenter_checkpoint_0.20.pth.tar',  #
        # 'helen': './ckpt/_train_celebA-face_20200225T1901/segmenter_checkpoint_0.14.pth.tar',  #
        # 'helen': './ckpt/_train_helen_20200225T2313/segmenter_checkpoint_0.14.pth.tar',  #pretrained by celeA
        # 'helen': './ckpt/_train_helen_20200226T1234/segmenter_checkpoint_0.11.pth.tar',  #pretrained by celeA
        # 'helen': './ckpt/_train_helen_20200226T1723/segmenter_checkpoint_0.10.pth.tar',  #pretrained by celeA loss:0.098
        # 'helen': './ckpt/_train_helen_20200226T2358/segmenter_checkpoint_0.36.pth.tar',  #  tain single
        # 'helen': './ckpt/_train_helen_20200227T1143/segmenter_checkpoint_0.43.pth.tar',  #  tain single batchsize 8
        # 'helen': './ckpt/_train_helen_20200227T1302/segmenter_checkpoint_0.35.pth.tar',  #  tain single batchsize 8
        # 'helen': './ckpt/_train_helen_20200227T1404/segmenter_checkpoint_0.40.pth.tar',  #  tain single batchsize 8
        # 'helen': './ckpt/_train_helen_20200227T1537/segmenter_checkpoint_0.35.pth.tar',  #  tain single batchsize 8
        # 'helen': './ckpt/_train_helen_20200227T1746/segmenter_checkpoint_0.33.pth.tar',  #  tain single batchsize 8
        # 'helen': './ckpt/_train_helen_20200227T1925/segmenter_checkpoint_0.35.pth.tar',  #  tain single batchsize 8
        # 'helen': './ckpt/_train_helen_20200227T2046/segmenter_checkpoint_0.37.pth.tar',  #  tain single batchsize 8
        # 'helen': './ckpt/_train_helen_20200228T1957/segmenter_checkpoint_0.31.pth.tar',  #  tain single batchsize 8
        # 'helen': './ckpt/_train_helen_20200228T2101/segmenter_checkpoint_0.30.pth.tar',  #  tain single batchsize 8 the best 90.3
        # 'helen': './ckpt/_train_helen_20200228T1200/segmenter_checkpoint_0.27.pth.tar',  #
        # 'helen': './ckpt/_train_helen_20200228T2310/segmenter_checkpoint_0.34.pth.tar',  #
        # 'helen': './ckpt/_train_helen_20200229T1700/segmenter_checkpoint_0.35.pth.tar',  #
        # 'helen': './ckpt/_train_helen_20200229T1940/segmenter_checkpoint_0.22.pth.tar',  #
        # 'helen': './ckpt/_train_helen_20200301T1545/segmenter_checkpoint_0.28.pth.tar',  #
        # 'helen': './ckpt/_train_helen_20200301T2109/segmenter_checkpoint_0.34.pth.tar',  #
        # 'helen': './ckpt/_train_helen_20200302T1943/segmenter_checkpoint_0.32.pth.tar',  #
        # 'helen': './ckpt/_train_helen_20200302T2201/segmenter_checkpoint_0.27.pth.tar',  #
        # 'helen': './ckpt/_train_helen_20200303T1147/segmenter_checkpoint_0.26.pth.tar',  #
        'helen': './ckpt/_train_helen_20200303T2336/segmenter_checkpoint_0.25.pth.tar',  #
        # 'helen_nohair': './ckpt/_train_helen_nohair_20200303T1416/segmenter_checkpoint_0.19.pth.tar',  #
        'helen_nohair': './ckpt/_train_helen_nohair_20200303T1646/segmenter_checkpoint_0.14.pth.tar',  #
        # 'celebA-face': './ckpt/_train_celebA-face_20200225T1518/segmenter_checkpoint_0.20.pth.tar',
        # 'celebA-face': './ckpt/_train_celebA-face_20200225T1901/segmenter_checkpoint_0.14.pth.tar', # perfect performance in celebA-face
        'celebA-face': './ckpt/_train_helen_20200226T1234/segmenter_checkpoint_0.11.pth.tar', #  test celeb-A with re-trained model by helen
        # no pre-trained mobilenetV2 poor performance
    }

# decoder_config = [[0, [0, 0, 5, 6], [4, 3, 5, 5], [2, 7, 2, 5]], [[3, 3], [2, 3], [4, 0]]]
# decoder_config = [[5, [0, 0, 5, 1], [4, 0, 8, 7], [6, 3, 3, 2]], [[1, 1], [3, 3], [2, 0]]]
# decoder_config = [[5, [0, 0, 3, 10], [3, 3, 7, 7], [7, 4, 7, 1]], [[0, 0], [2, 0], [0, 1]]] #0.7035
# [[1, [1, 1, 5, 9], [2, 1, 9, 1], [3, 6, 1, 9]], [[1, 3], [1, 0], [4, 2]]] #0.7040 (reward)
# [[6, [0, 1, 2, 5], [0, 2, 5, 3], [6, 5, 1, 10]], [[1, 3], [0, 3], [3, 4]]]  #0.7108 all cls
# decoder_config = [[1, [0, 0, 6, 0], [0, 3, 8, 3], [3, 0, 6, 3]], [[1, 2], [2, 4], [0, 4]]] #0.7137 all cls
decoder_config = \
    {
        # 'celebA':[[5, [1, 0, 3, 5], [1, 0, 10, 10], [6, 6, 0, 10]], [[1, 0], [4, 2], [3, 2]]],  # 0.803
        # 'celebA':[[5, [1, 0, 3, 5], [1, 0, 10, 10], [6, 6, 0, 10]], [[1, 0], [4, 2], [3, 2], [0,2], [1,4]]],  # 0.803
        # 'celebA': [[1, [1, 1, 5, 5], [1, 0, 2, 7], [1, 4, 7, 5]], [[2, 1], [3, 0], [3, 2]]] ,  #
        'celebA':  [[3, [1, 1, 5, 0], [0, 4, 1, 9], [4, 3, 2, 0]], [[3, 3], [2, 1], [2, 0], [1,4]]] ,  #
        # 'celebA':  [[1, [0, 0, 5, 1], [3, 2, 1, 10], [3, 5, 10, 9]], [[2, 3], [1, 2], [2, 2]]] ,  #
        # 'celebA':  [[9, [1, 0, 1, 1], [3, 3, 0, 0], [7, 6, 4, 2]], [[3, 3], [3, 4], [5, 5]]] ,  #
        #'EG1800':[[1, [0, 0, 10, 9], [0, 1, 2, 7], [2, 0, 0, 9]], [[2, 0], [3, 2], [2, 4]]], #0.9636 EG1800
        # 'EG1800': [[2, [1, 0, 10, 8], [2, 3, 1, 8], [2, 1, 2, 2]], [[3, 1], [2, 4], [5, 5]]],
        'EG1800':[[1, [0, 0, 10, 9], [0, 1, 2, 7], [2, 0, 0, 9]], [[2, 0], [3, 2], [2, 4]]], #0.9636 EG1800:
        'celebA-binary':[[1, [0, 0, 10, 9], [0, 1, 2, 7], [2, 0, 0, 9]], [[2, 0], [3, 2], [2, 4]]], #0.9636 EG1800:
        # 'helen':[[5, [1, 0, 3, 5], [1, 0, 10, 10], [6, 6, 0, 10]], [[1, 0], [4, 2], [3, 2]]],
        'helen':[[5, [1, 0, 3, 5], [1, 0, 10, 10], [6, 6, 0, 10]], [[1, 0], [4, 2], [3, 2],[0,2],[1,4]]], #cur the best
        'helen_nohair':[[5, [1, 0, 3, 5], [1, 0, 10, 10], [6, 6, 0, 10]], [[1, 0], [4, 2], [3, 2],[0,2],[1,4]]], #cur the best
        # 'helen':[[5, [1, 0, 3, 5], [1, 0, 10, 10], [6, 6, 0, 10]], [[1, 0], [4, 2], [3, 2],[0,2],[1,4],[0,3]]], #0.89
        # 'helen':[[5, [1, 0, 3, 5], [1, 0, 10, 10], [6, 6, 0, 10]], [[1, 0], [4, 2], [3, 2],[0,2],[1,4],[1,5]]], #0.89
        # 'helen':[[5, [1, 0, 3, 5], [1, 0, 10, 10], [6, 6, 0, 10]], [[1, 0], [4, 2], [3, 2],[0,2]]],
        'celebA-face': [[5, [1, 0, 3, 5], [1, 0, 10, 10], [6, 6, 0, 10]], [[1, 0], [4, 2], [3, 2]]],
    }
# decoder_config = [[10, [1, 0, 8, 10], [0, 1, 3, 2], [7, 1, 4, 3]], [[3, 0], [3, 4], [3, 2]]] #0.095 worst all cls
# [[10, [1, 1, 5, 2], [3, 0, 3, 4], [6, 7, 5, 9]], [[0, 0], [4, 3], [3, 1]]] #0.1293 all cls
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="NAS Search")

    parser.add_argument("--dataset_type", type=str, default= 'celebA', #'helen',#'celebA-binary',#'EG1800',
                        help="dataset type to be trained or valued.")

    # Dataset
    # parser.add_argument("--train-dir", type=str, default=TRAIN_DIR,
    #                     help="Path to the training set directory.")
    # parser.add_argument("--val-dir", type=str, default=VAL_DIR,
    #                     help="Path to the validation set directory.")
    # parser.add_argument("--train-list", type=str, default=TRAIN_LIST,
    #                     help="Path to the training set list.")
    # parser.add_argument("--val-list", type=str, default=VAL_LIST,
    #                     help="Path to the validation set list.")
    parser.add_argument("--meta-train-prct", type=int, default=META_TRAIN_PRCT,
                        help="Percentage of examples for meta-training set.")
    parser.add_argument("--shorter-side", type=int, nargs='+', default=SHORTER_SIDE,
                        help="Shorter side transformation.")
    parser.add_argument("--crop-size", type=int, nargs='+', default=CROP_SIZE,
                        help="Crop size for training,")
    parser.add_argument("--normalise-params", type=list, default=NORMALISE_PARAMS,
                        help="Normalisation parameters [scale, mean, std],")
    parser.add_argument("--batch-size", type=int, nargs='+', default=BATCH_SIZE,
                        help="Batch size to train the segmenter model.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of workers for pytorch's dataloader.")
    # parser.add_argument("--num-classes", type=int, nargs='+', default=NUM_CLASSES,
    #                     help="Number of output classes for each task.")
    parser.add_argument("--low-scale", type=float, default=LOW_SCALE,
                        help="Lower bound for random scale")
    parser.add_argument("--high-scale", type=float, default=HIGH_SCALE,
                        help="Upper bound for random scale")
    # parser.add_argument("--n-task0", type=int, default=N_TASK0,
    #                     help="Number of images per task0 (trainval)")
    parser.add_argument("--val-shorter-side", type=int, default=VAL_SHORTER_SIDE,
                        help="Shorter side transformation during validation.")
    parser.add_argument("--val-crop-size", type=int, default=VAL_CROP_SIZE,
                        help="Crop size for validation.")
    parser.add_argument("--val-batch-size", type=int, default=VAL_BATCH_SIZE,
                        help="Batch size to validate the segmenter model.")

    # Encoder
    parser.add_argument('--enc-grad-clip', type=float, default=ENC_GRAD_CLIP,
                        help="Clip norm of encoder gradients to this value.")

    # Decoder
    parser.add_argument('--dec-grad-clip', type=float, default=DEC_GRAD_CLIP,
                        help="Clip norm of decoder gradients to this value.")
    parser.add_argument('--dec-aux-weight', type=float, default=DEC_AUX_WEIGHT,
                        help="Auxiliary loss weight for each aggregate head.")

    # General
    parser.add_argument("--freeze-bn", type=bool, nargs='+', default=FREEZE_BN,
                        help='Whether to keep batch norm statistics intact.')
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help='Number of epochs to train for the controller.')
    parser.add_argument("--num-segm-epochs", type=int, nargs='+', default=NUM_SEGM_EPOCHS,
                        help='Number of epochs to train for each sampled network.')
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY,
                        help='Print information every often.')
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help='Seed to provide (near-)reproducibility.')
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Path to directory for storing checkpoints.")
    parser.add_argument("--ckpt-path", type=str, default=SEGMENTER_CKPT_PATH,
                        help="Path to the checkpoint file.")
    parser.add_argument("--val-every", nargs='+', type=int, default=VAL_EVERY,
                        help="How often to validate current architecture.")
    parser.add_argument("--summary-dir", type=str, default=SUMMARY_DIR,
                        help="Summary directory.")
    # Controller
    parser.add_argument("--hidden_size", type=int, default=100,
                        help="Number of neurons in the controller's RNN.")
    parser.add_argument("--num_lstm_layers", type=int, default=2,
                        help="Number of layers in the controller.")
    parser.add_argument("--op-size", type=int, default=OP_SIZE,
                        help="Number of unique operations.")
    parser.add_argument("--agg-cell-size", type=int, default=AGG_CELL_SIZE,
                        help="Common size inside decoder")
    parser.add_argument("--bl-dec", type=float, default=BL_DEC,
                        help="Baseline decay.")
    parser.add_argument("--agent-ctrl", type=str, default=AGENT_CTRL,
                        help="Gradient estimator algorithm")
    parser.add_argument("--num-cells", type=int, default=NUM_CELLS,
                        help="Number of cells to apply.")
    parser.add_argument("--num-branches", type=int, default=NUM_BRANCHES,
                        help="Number of branches inside the learned cell.")
    parser.add_argument("--aux-cell", type=bool, default=AUX_CELL,
                        help="Whether to use the cell design in-place of auxiliary cell.")
    parser.add_argument("--sep-repeats", type=int, default=SEP_REPEATS,
                        help="Number of repeats inside Sep Convolution.")

    return parser.parse_args()


class Segmenter(nn.Module):
    """Create Segmenter"""
    def __init__(self, encoder, decoder):
        super(Segmenter, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

### see the color config in dmh code
color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
'''imgs = [
    '../examples/face_test_img/2345.jpg',
    '../examples/face_test_img/3456.jpg',
    '../examples/face_test_img/6789.jpg',
]'''

# TEST_IMG_PATH = {'celebA':'../data/datasets/portrait_parsing','EG1800':'../data/datasets/portrait_seg/EG1800'}
# RAW_IMAGE_PATH = {'celebA':'CelebA-HA-img-resize','EG1800':'Images','celebA-binary':'CelebA-HA-img-resize'}
# MASK_IMAGE_PATH = {'celebA':'CelebAMask-HQ-mask','EG1800':'Labels','celebA-binary':'CelebAMask-HQ-mask-binary'}
def main():
    # Set-up experiment
    args = get_arguments()
    logger = logging.getLogger(__name__)
    exp_name = time.strftime('%H_%M_%S')
    # dir_name = '{}/{}'.format(args.summary_dir, exp_name)
    # if not os.path.exists(dir_name):
    #     os.makedirs(dir_name)
    # arch_writer = open('{}/genotypes.out'.format(dir_name), 'w')
    logger.info(" Running Experiment {}".format(exp_name))
    args.num_tasks = len(NUM_CLASSES[args.dataset_type])
    segm_crit = nn.NLLLoss2d(ignore_index=255).cuda()

    # Set-up random seeds
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Initialise encoder
    encoder = create_encoder()
    logger.info(" Loaded Encoder with #TOTAL PARAMS={:3.2f}M"
                .format(compute_params(encoder)[0] / 1e6))
    def create_segmenter(encoder):
        with torch.no_grad():
            #decoder_config, entropy, log_prob = agent.controller.sample()
            #stub the decoder arch

            decoder = Decoder(inp_sizes=encoder.out_sizes,
                              num_classes=NUM_CLASSES[args.dataset_type][0],
                              config=decoder_config[args.dataset_type],
                              agg_size=args.agg_cell_size,
                              aux_cell=args.aux_cell,
                              repeats=args.sep_repeats)

        # Fuse encoder and decoder
        segmenter = nn.DataParallel(Segmenter(encoder, decoder)).cuda()
        logger.info(" Created Segmenter, #PARAMS (Total, No AUX)={}".format(
            compute_params(segmenter)
        ))
        return segmenter

    # Sample first configuration
    segmenter  = create_segmenter(encoder)
    del encoder

    #NUM_CLASSES = [17, 17]
    segmenter.load_state_dict(torch.load(args.ckpt_path[args.dataset_type]))
    logger.info(" Loaded Encoder with #TOTAL PARAMS={:3.2f}M"
                .format(compute_params(segmenter)[0] / 1e6))

    segmenter.eval()

    TEST_NUM =  6 #4
    fig, axes = plt.subplots(3, TEST_NUM, figsize=(12, 12))
    # axes.set_xticks
    ax= axes.ravel()
    color_array = np.array(color_list)
    random.seed()

    # if args.dataset_type == 'celebA' :
    #     imgs = [os.path.join(TEST_IMG_PATH['celebA'],RAW_IMAGE_PATH['celebA'],'{}.jpg'.format(random.randint(0,30000))) for i in range(TEST_NUM)]
    #     msks = [imgs[i].replace(RAW_IMAGE_PATH['celebA'],MASK_IMAGE_PATH['celebA']).replace('jpg','png') for i in range(TEST_NUM)]
    # elif args.dataset_type == 'EG1800' :#or args.dataset_type == 'celebA-binary':
    #     imgs = [os.path.join(TEST_IMG_PATH['EG1800'],RAW_IMAGE_PATH['EG1800'],'{}'.format(random.randint(0,100)).rjust(5,'0')+'.png') for i in range(TEST_NUM)]
    #     msks = [imgs[i].replace(RAW_IMAGE_PATH['EG1800'],MASK_IMAGE_PATH['EG1800']) for i in range(TEST_NUM)]


    data_file=dataset_dirs[args.dataset_type]['VAL_LIST']
    data_dir=dataset_dirs[args.dataset_type]['VAL_DIR']

    with open(data_file, 'rb') as f:
        datalist = f.readlines()
    try:
        datalist = [
            (k, v) for k, v, _ in \
                map(lambda x: x.decode('utf-8').strip('\n').split('\t'), datalist)]
    except ValueError: # Adhoc for test.
        datalist = [
            (k, k) for k in map(lambda x: x.decode('utf-8').strip('\n'), datalist)]

    random_array = random.sample(range(0,len(datalist)),TEST_NUM)
    imgs = [os.path.join(data_dir,datalist[i][0]) for i in random_array]
    msks = [os.path.join(data_dir,datalist[i][1]) for i in random_array]

    imgs_all = [os.path.join(data_dir,datalist[i][0]) for i in range(0,len(datalist))]
    msks_all = [os.path.join(data_dir,datalist[i][1]) for i in range(0,len(datalist))]

    '''imgs = [
        # '../data/datasets/EG1800/Images/02323.png', # EG1800
        # '../data/datasets/EG1800/Images/01232.png',
        # '../data/datasets/EG1800/Images/02178.png',
        # '../data/datasets/EG1800/Images/02033.png',
        # '../data/datasets/EG1800/Images/02235.png',
        # '../data/datasets/EG1800/Images/00105.png',
        # '../data/datasets/EG1800/Images/00105.png',
        '../data/datasets/EG1800/00009_224.png',
        # '../data/datasets/helen/test/141794264_1_image.jpg',   #HELEN
        # '../data/datasets/helen/test/107635070_1_image.jpg',
        # '../data/datasets/helen/test/1030333538_1_image.jpg',
        # '../data/datasets/helen/test/122276700_1_image.jpg',
        # '../data/datasets/helen/test/1344304961_1_image.jpg',
        # '../data/datasets/helen/test/1240746154_1_image.jpg',
        # '../data/datasets/celebA/CelebA-HA-img-resize/29044.jpg',  #celebA
        # '../data/datasets/celebA/CelebA-HA-img-resize/27039.jpg',
        # '../data/datasets/celebA/CelebA-HA-img-resize/27047.jpg',
        # '../data/datasets/celebA/CelebA-HA-img-resize/27037.jpg',
        # '../data/datasets/celebA/CelebA-HA-img-resize/29045.jpg',
        # '../data/datasets/celebA/CelebA-HA-img-resize/29022.jpg',
        # '../data/datasets/celebA/CelebA-HA-img-resize/29312.jpg',  #celebA-generilize
        # '../data/datasets/celebA/CelebA-HA-img-resize/27039.jpg',
        # '../data/datasets/celebA/CelebA-HA-img-resize/29085.jpg',
        # '../data/datasets/celebA/CelebA-HA-img-resize/29068.jpg',
        # '../data/datasets/celebA/CelebA-HA-img-resize/29039.jpg',
    ]
    msks = [
        # '../data/datasets/EG1800/Labels/02323.png',  # EG1800
        # '../data/datasets/EG1800/Labels/01232.png',
        # '../data/datasets/EG1800/Labels/02178.png',
        # '../data/datasets/EG1800/Labels/02033.png',
        # '../data/datasets/EG1800/Labels/02235.png',
        # '../data/datasets/EG1800/Labels/00105.png',
        '../data/datasets/EG1800/00009_224_mask.png',
        # '../data/datasets/helen/test/141794264_1_label.png',  # HELEN
        # '../data/datasets/helen/test/107635070_1_label.png',
        # '../data/datasets/helen/test/1030333538_1_label.png',
        # '../data/datasets/helen/test/122276700_1_label.png',
        # '../data/datasets/helen/test/1344304961_1_label.png',
        # '../data/datasets/helen/test/1240746154_1_label.png'
        # '../data/datasets/celebA/CelebAMask-HQ-mask-all-class/29044.png',  #celebA
        # '../data/datasets/celebA/CelebAMask-HQ-mask-all-class/27039.png',
        # '../data/datasets/celebA/CelebAMask-HQ-mask-all-class/27047.png',
        # '../data/datasets/celebA/CelebAMask-HQ-mask-all-class/27037.png',
        # '../data/datasets/celebA/CelebAMask-HQ-mask-all-class/29045.png',
        # '../data/datasets/celebA/CelebAMask-HQ-mask-all-class/29022.png',
        # '../data/datasets/celebA/CelebAMask-HQ-mask-all-class/29312.png',  #celebA -generilize
        # '../data/datasets/celebA/CelebAMask-HQ-mask-all-class/27039.png',
        # '../data/datasets/celebA/CelebAMask-HQ-mask-all-class/29085.png',
        # '../data/datasets/celebA/CelebAMask-HQ-mask-all-class/29068.png',
        # '../data/datasets/celebA/CelebAMask-HQ-mask-all-class/29039.png',
    ]'''

    show_raw_portrait_seg = 1
    for i,img_path in enumerate(imgs):
        logger.info("Testing image:{}".format(img_path))
        img = np.array(Image.open(img_path))
        msk = np.array(Image.open(msks[i]))
        orig_size = img.shape[:2][::-1]
        ax[i].imshow(img,aspect='auto')
        plt.axis('off')
        if args.dataset_type =='EG1800' and show_raw_portrait_seg:
            img_msk = img.copy()
            img_msk[msk == 0] = (0,0,255)
            ax[TEST_NUM+i].imshow(img_msk,aspect='auto')
        elif args.dataset_type == 'helen' or args.dataset_type == 'helen_nohair':
            ax[TEST_NUM+i].imshow(img,aspect='auto')
            msk = color_array[msk]
            ax[TEST_NUM+i].imshow(msk,aspect='auto',alpha=0.7)
        else:
            # ax[3*i+1].imshow(img,aspect='auto')
            msk = color_array[msk]
            ax[TEST_NUM+i].imshow(msk,aspect='auto',)

        plt.axis('off')

        img_inp = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float().to(device)
        segm = segmenter(img_inp)[0].squeeze().data.cpu().numpy().transpose((1, 2, 0)) #47*63*21
        #cal params and flops
        # input = torch.randn(1,3,512,512)
        flops, params = profile(segmenter, inputs = (img_inp,), )
        flops, params = clever_format([flops, params], "%.3f")
        print(flops)
        print(params)
        segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC) #375*500*21
        segm = segm.argmax(axis=2).astype(np.uint8)
        if args.dataset_type =='EG1800'  and show_raw_portrait_seg:
            img_segm = img.copy()
            img_segm[segm == 0] = (0,0,255)
            ax[2*TEST_NUM+i].imshow(img_segm,aspect='auto')
        elif args.dataset_type == 'helen' or args.dataset_type == 'helen_nohair':
            segm = color_array[segm]  #375*500*3  #wath this usage ,very very important
            ax[2*TEST_NUM+i].imshow(img,aspect='auto')
            ax[2*TEST_NUM+i].imshow(segm,aspect='auto',alpha=0.7)
            # print(segm.shape)
        else:
            segm = color_array[segm]  # 375*500*3  #wath this usage ,very very important
            # ax[3 * i + 2].imshow(img, aspect='auto')
            ax[2*TEST_NUM+i].imshow(segm, aspect='auto',)
            ax[2*TEST_NUM+i].set_xticks([])
            ax[2*TEST_NUM+i].set_yticks([])
        plt.axis('off')
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()
    # fig.savefig('./eg1800.jpg')

    if args.dataset_type == 'helen' or args.dataset_type == 'helen_nohair' or args.dataset_type == 'celebA':
        validate_output_dir = os.path.join(dataset_dirs[args.dataset_type]['VAL_DIR'], 'validate_output')
        validate_gt_dir = os.path.join(dataset_dirs[args.dataset_type]['VAL_DIR'], 'validate_gt')
        validate_color_dir = os.path.join(dataset_dirs[args.dataset_type]['VAL_DIR'], 'validate_output_color')

        if not os.path.exists(validate_output_dir):
            os.makedirs(validate_output_dir)
        else:
            shutil.rmtree(validate_output_dir)
            os.makedirs(validate_output_dir)

        if not os.path.exists(validate_gt_dir):
            os.makedirs(validate_gt_dir)
        else:
            shutil.rmtree(validate_gt_dir)
            os.makedirs(validate_gt_dir)

        if not os.path.exists(validate_color_dir):
            os.makedirs(validate_color_dir)
        else:
            shutil.rmtree(validate_color_dir)
            os.makedirs(validate_color_dir)

        # save_color = 0
        for i, img_path in enumerate(imgs_all):
            # logger.info("Testing image:{}".format(img_path))
            img = np.array(Image.open(img_path))
            msk = np.array(Image.open(msks_all[i]))
            orig_size = img.shape[:2][::-1]

            img_inp = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float().to(device)
            segm = segmenter(img_inp)[0].squeeze().data.cpu().numpy().transpose((1, 2, 0))  # 47*63*21
            if args.dataset_type == 'celebA':
                # msk = cv2.resize(msk,segm.shape[0:2],interpolation=cv2.INTER_NEAREST)
                segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)  # 375*500*21
            else:
                segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)  # 375*500*21
            segm = segm.argmax(axis=2).astype(np.uint8)

            image_name = img_path.split('/')[-1].split('.')[0]
            # image_name = val_loader.dataset.datalist[i][0].split('/')[1].split('.')[0]
            cv2.imwrite(os.path.join(validate_color_dir, "{}.png".format(image_name)), color_array[segm])
            # cv2.imwrite(os.path.join(validate_gt_dir, "{}.png".format(image_name)), color_array[msk])
            cv2.imwrite(os.path.join(validate_output_dir, "{}.png".format(image_name)), segm)
            cv2.imwrite(os.path.join(validate_gt_dir, "{}.png".format(image_name)), msk)

        if args.dataset_type == 'celebA':
            cal_f1_score_celebA(validate_gt_dir,validate_output_dir) # temp comment
            # pass
        else:
            cal_f1_score(validate_gt_dir,validate_output_dir)

        plt.show()
    else: # EG1800
        # Create dataloaders
        _, val_loader, _ = create_loaders(args)
        try:
            val_loader.dataset.set_stage('val')
        except AttributeError:
            val_loader.dataset.dataset.set_stage('val')  # for subseta

        task_miou = validate(segmenter,
                         val_loader,
                         1,
                         1, #[5,1]
                         num_classes=NUM_CLASSES[args.dataset_type][0],
                         print_every=args.print_every)

if __name__ == '__main__':
    main()
