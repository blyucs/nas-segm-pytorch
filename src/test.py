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

from utils.helpers import prepare_img
os.environ["CUDA_VISIBLE_DEVICES"]="0"
logging.basicConfig(level=logging.INFO)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cmap = np.load('./utils/cmap.npy')
SEGMENTER_CKPT_PATH = \
    {
        'celebA':'./ckpt/20200111T1841/segmenter_checkpoint.pth.tar',
        #'EG1800':'./ckpt/train20200117T1958/segmenter_checkpoint.pth.tar'
        'EG1800':'./ckpt/train20200118T1128/segmenter_checkpoint.pth.tar'  # 00079,00094,good, the best model currently
       # 'EG1800':'./ckpt/train20200118T1224/segmenter_checkpoint.pth.tar'
        #'EG1800':'./ckpt/train20200118T1239/segmenter_checkpoint.pth.tar'
}

# decoder_config = [[0, [0, 0, 5, 6], [4, 3, 5, 5], [2, 7, 2, 5]], [[3, 3], [2, 3], [4, 0]]]
# decoder_config = [[5, [0, 0, 5, 1], [4, 0, 8, 7], [6, 3, 3, 2]], [[1, 1], [3, 3], [2, 0]]]
# decoder_config = [[5, [0, 0, 3, 10], [3, 3, 7, 7], [7, 4, 7, 1]], [[0, 0], [2, 0], [0, 1]]] #0.7035
# [[1, [1, 1, 5, 9], [2, 1, 9, 1], [3, 6, 1, 9]], [[1, 3], [1, 0], [4, 2]]] #0.7040 (reward)
# [[6, [0, 1, 2, 5], [0, 2, 5, 3], [6, 5, 1, 10]], [[1, 3], [0, 3], [3, 4]]]  #0.7108 all cls
# decoder_config = [[1, [0, 0, 6, 0], [0, 3, 8, 3], [3, 0, 6, 3]], [[1, 2], [2, 4], [0, 4]]] #0.7137 all cls
decoder_config = \
    {
        'celebA':[[5, [1, 0, 3, 5], [1, 0, 10, 10], [6, 6, 0, 10]], [[1, 0], [4, 2], [3, 2]]],  # 0.803
        'EG1800':[[1, [0, 0, 10, 9], [0, 1, 2, 7], [2, 0, 0, 9]], [[2, 0], [3, 2], [2, 4]]] #0.9636 EG1800
    }
# decoder_config = [[10, [1, 0, 8, 10], [0, 1, 3, 2], [7, 1, 4, 3]], [[3, 0], [3, 4], [3, 2]]] #0.095 worst all cls
# [[10, [1, 1, 5, 2], [3, 0, 3, 4], [6, 7, 5, 9]], [[0, 0], [4, 3], [3, 1]]] #0.1293 all cls
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="NAS Search")

    parser.add_argument("--dataset_type", type=str, default='EG1800',#'EG1800',
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


color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
'''imgs = [
    '../examples/face_test_img/2345.jpg',
    '../examples/face_test_img/3456.jpg',
    '../examples/face_test_img/6789.jpg',
]'''

TEST_IMG_PATH = {'celebA':'../data/datasets/celebA','EG1800':'../data/datasets/portrait_seg/EG1800'}
RAW_IMAGE_PATH = {'celebA':'CelebA-HA-img-resize','EG1800':'Images'}
MASK_IMAGE_PATH = {'celebA':'CelebAMask-HQ-mask','EG1800':'Labels'}
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



    TEST_NUM = 3
    fig, axes = plt.subplots(TEST_NUM, 3, figsize=(12, 12))
    ax= axes.ravel()
    color_array = np.array(color_list)
    random.seed()

    if(args.dataset_type == 'celebA'):
        imgs = [os.path.join(TEST_IMG_PATH['celebA'],RAW_IMAGE_PATH['celebA'],'{}.jpg'.format(random.randint(0,30000))) for i in range(TEST_NUM)]
        msks = [imgs[i].replace(RAW_IMAGE_PATH['celebA'],MASK_IMAGE_PATH['celebA']).replace('jpg','png') for i in range(TEST_NUM)]
    elif(args.dataset_type == 'EG1800'):
        imgs = [os.path.join(TEST_IMG_PATH['EG1800'],RAW_IMAGE_PATH['EG1800'],'{}'.format(random.randint(0,100)).rjust(5,'0')+'.png') for i in range(TEST_NUM)]
        msks = [imgs[i].replace(RAW_IMAGE_PATH['EG1800'],MASK_IMAGE_PATH['EG1800']) for i in range(TEST_NUM)]

    '''
    imgs = [
        '../data/datasets/face_seg_dataset/rawimage/212409770_1.jpg',

        '../data/datasets/face_seg_dataset/rawimage/114226877_1.jpg'
    ]
    msks = [
        '../data/datasets/face_seg_dataset/class_labels/212409770_1.png',
        '../data/datasets/face_seg_dataset/class_labels/114226877_1.png'
    ]
    '''
    for i,img_path in enumerate(imgs):
        logger.info("Testing image:{}".format(img_path))
        img = np.array(Image.open(img_path))
        msk = np.array(Image.open(msks[i]))
        msk = color_array[msk]
        orig_size = img.shape[:2][::-1]
        ax[3*i].imshow(img,aspect='auto')
        ax[3*i+1].imshow(msk,aspect='auto')

        img_inp = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float().to(device)
        segm = segmenter(img_inp)[0].squeeze().data.cpu().numpy().transpose((1, 2, 0)) #47*63*21
        segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC) #375*500*21
        segm = color_array[segm.argmax(axis=2).astype(np.uint8)]  #375*500*3  #wath this usage ,very very important
        # print(segm.shape)
        ax[3*i+2].imshow(segm,aspect='auto')
    plt.show()

if __name__ == '__main__':
    main()
