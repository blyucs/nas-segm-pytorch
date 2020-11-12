"""Main file for search.

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
import json
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
from utils.f1_score import *
from utils.solvers import create_optimisers
import matplotlib.pyplot as plt
import shutil
import cv2

from utils.helpers import prepare_img
os.environ["CUDA_VISIBLE_DEVICES"]="3"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.basicConfig(level=logging.INFO)
# TRAIN_EPOCH_NUM = {'celebA':[40,10],'EG1800':[0,50],'celebA-binary':[0,6]}

SEGMENTER_CKPT_PATH = \
    {
        'celebA':'./ckpt/20200111T1841/segmenter_checkpoint.pth.tar',
        #'EG1800':'./ckpt/train20200117T1958/segmenter_checkpoint.pth.tar'
        #'EG1800':'./ckpt/train20200118T1128/segmenter_checkpoint.pth.tar' , # MIOU 0.924 in EG1800, currently best(0217)

        #'EG1800': './ckpt/_train_celebA-binary_20200118T1715/segmenter_checkpoint.pth.tar', #MIOU 0.976 in cele-binary ,currently best(0217)
        #'EG1800':'./ckpt/_train_EG1800_20200217T1059/segmenter_checkpoint.pth.tar',
        #'EG1800':'./ckpt/train20200118T1224/segmenter_checkpoint.pth.tar'
        #'EG1800':'./ckpt/train20200118T1239/segmenter_checkpoint.pth.tar'
        #'EG1800': './ckpt/_train_EG1800_20200217T1405/segmenter_checkpoint.pth.tar',# [2,[1,0,10,8]]
        'celebA-binary': './ckpt/_train_celebA-binary_20200118T1715/segmenter_checkpoint.pth.tar', #MIOU 0.976 in cele-binary ,0.944 in EG1800,currently best(0217)
        # 'EG1800': './ckpt/_train_EG1800_20200217T1922/segmenter_checkpoint.pth.tar',
        #'celebA-binary': './ckpt/_train_EG1800_20200217T1922/segmenter_checkpoint.pth.tar',
        #'EG1800': './ckpt/_train_EG1800_20200217T2216/segmenter_checkpoint.pth.tar',

        # 'EG1800': './ckpt/_train_EG1800_20200218T1319/segmenter_checkpoint.pth.tar',
        # 'EG1800': './ckpt/_train_EG1800_20200218T1503/segmenter_checkpoint.pth.tar',
        # 'EG1800': './ckpt/_train_EG1800_20200218T1606/segmenter_checkpoint.pth.tar',
        # 'EG1800': './ckpt/_train_EG1800_20200218T1842/segmenter_checkpoint.pth.tar',
        'EG1800': './ckpt/_train_EG1800_20200218T2034/segmenter_checkpoint.pth.tar', #0.967
        # 'EG1800': './ckpt/_train_EG1800_20200218T2158/segmenter_checkpoint.pth.tar',  # 0.873
        # 'helen': './ckpt/_train_helen_20200223T1724/segmenter_checkpoint.pth.tar',  #
        # 'helen': './ckpt/_train_helen_20200224T1611/segmenter_checkpoint.pth.tar',  # 0.81
        # 'helen': './ckpt/_train_helen_20200225T1251/segmenter_checkpoint.pth.tar',  # 0.873
        # 'helen': './ckpt/_train_helen_20200225T1319/segmenter_checkpoint.pth.tar',  # 0.7476  # no pre-trained mobilenetV2 poor performance
        # 'helen': './ckpt/_train_celebA-face_20200225T1518/segmenter_checkpoint_0.20.pth.tar',  #0.74176
        # 'helen': './ckpt/_train_celebA-face_20200225T1901/segmenter_checkpoint_0.14.pth.tar',
        # 'helen': './ckpt/_train_helen_20200225T2313/segmenter_checkpoint_0.14.pth.tar',
        # 'helen': './ckpt/_train_helen_20200226T1234/segmenter_checkpoint_0.11.pth.tar',  #pretrained by celeA
        # 'helen': './ckpt/_train_helen_20200226T1723/segmenter_checkpoint_0.10.pth.tar',  # pretrained by celeA

        'helen': './ckpt/_train_helen_20200226T2358/segmenter_checkpoint_0.36.pth.tar',
        # pretrained by celeA loss:0.098
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
        'EG1800':[[1, [0, 0, 10, 9], [0, 1, 2, 7], [2, 0, 0, 9]], [[2, 0], [3, 2], [2, 4]]], #0.924
        #'EG1800': [[2, [1, 0, 10, 8], [2, 3, 1, 8], [2, 1, 2, 2]], [[3, 1], [2, 4], [5, 5]]],
        'celebA-binary':[[1, [0, 0, 10, 9], [0, 1, 2, 7], [2, 0, 0, 9]], [[2, 0], [3, 2], [2, 4]]], #0.976
        'helen': [[5, [1, 0, 3, 5], [1, 0, 10, 10], [6, 6, 0, 10]], [[1, 0], [4, 2], [3, 2]]],
    }
# decoder_config = [[10, [1, 0, 8, 10], [0, 1, 3, 2], [7, 1, 4, 3]], [[3, 0], [3, 4], [3, 2]]] #0.095 worst all cls
# [[10, [1, 1, 5, 2], [3, 0, 3, 4], [6, 7, 5, 9]], [[0, 0], [4, 3], [3, 1]]] #0.1293 all cls


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="NAS Search")

    parser.add_argument("--dataset_type", type=str, default= 'EG1800',#'celebA-binary',
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
    # parser.add_argument("--batch-size", type=int, nargs='+', default=BATCH_SIZE,
    #                     help="Batch size to train the segmenter model.")
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
    # parser.add_argument("--val-batch-size", type=int, default=VAL_BATCH_SIZE,
    #                     help="Batch size to validate the segmenter model.")

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
    # parser.add_argument("--num-segm-epochs", type=int, nargs='+', default=NUM_SEGM_EPOCHS,
    #                     help='Number of epochs to train for each sampled network.')
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY,
                        help='Print information every often.')
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help='Seed to provide (near-)reproducibility.')
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Path to directory for storing checkpoints.")
    parser.add_argument("--val-every", nargs='+', type=int, default=VAL_EVERY,
                        help="How often to validate current architecture.")
    parser.add_argument("--summary-dir", type=str, default=SUMMARY_DIR,
                        help="Summary directory.")

    parser.add_argument("--ckpt-path", type=str, default=SEGMENTER_CKPT_PATH,
                        help="Path to the checkpoint file.")

    # Optimisers
    parser.add_argument("--lr-enc", type=float, nargs='+', default=LR_ENC,
                        help="Learning rate for encoder.")
    parser.add_argument("--lr-dec", type=float, nargs='+', default=LR_DEC,
                        help="Learning rate for decoder.")
    parser.add_argument("--lr-ctrl", type=float, default=LR_CTRL,
                        help="Learning rate for controller.")
    parser.add_argument("--mom-enc", type=float, nargs='+', default=MOM_ENC,
                        help="Momentum for encoder.")
    parser.add_argument("--mom-dec", type=float, nargs='+', default=MOM_DEC,
                        help="Momentum for decoder.")
    parser.add_argument("--mom-ctrl", type=float, default=MOM_CTRL,
                        help="Momentum for controller.")
    parser.add_argument("--wd-enc", type=float, nargs='+', default=WD_ENC,
                        help="Weight decay for encoder.")
    parser.add_argument("--wd-dec", type=float, nargs='+', default=WD_DEC,
                        help="Weight decay for decoder.")
    parser.add_argument("--wd-ctrl", type=float, default=WD_CTRL,
                        help="Weight decay rate for controller.")
    parser.add_argument("--optim-enc", type=str, default=OPTIM_ENC,
                        help="Optimiser algorithm for encoder.")
    parser.add_argument("--optim-dec", type=str, default=OPTIM_DEC,
                        help="Optimiser algorithm for decoder.")
    parser.add_argument("--do-kd", type=bool, default=DO_KD,
                        help="Whether to do knowledge distillation (KD).")
    parser.add_argument("--kd-coeff", type=float, default=KD_COEFF,
                        help="KD loss coefficient.")
    parser.add_argument("--do-polyak", type=bool, default=DO_POLYAK,
                        help="Whether to do Polyak averaging.")


    parser.add_argument("--agg-cell-size", type=int, default=AGG_CELL_SIZE,
                        help="Common size inside decoder")
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
    # args.num_tasks = len(NUM_CLASSES[args.dataset_type])
    # segm_crit = nn.NLLLoss2d(ignore_index=255).cuda()
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
    segmenter= create_segmenter(encoder)
    del encoder

    color_array = np.array(color_list)
    segmenter.load_state_dict(torch.load(args.ckpt_path[args.dataset_type]), strict=False)
    logger.info(" Loaded Encoder with #TOTAL PARAMS={:3.2f}M"
                .format(compute_params(segmenter)[0] / 1e6))

    # Create dataloaders
    _, val_loader, _ = create_loaders(args)
    try:
        val_loader.dataset.set_stage('val')
    except AttributeError:
        val_loader.dataset.dataset.set_stage('val')  # for subset
    if args.dataset_type == 'helen':
        validate_output_dir = os.path.join(dataset_dirs['helen']['VAL_DIR'], 'validate_output')
        validate_gt_dir = os.path.join(dataset_dirs['helen']['VAL_DIR'], 'validate_gt')
        validate_color_dir = os.path.join(dataset_dirs['helen']['VAL_DIR'], 'validate_output_color')
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
        _out_type_1_ = 0# helen validte type flag
        # save_color = 0
        for i,sample in enumerate(val_loader):
            for j in range(VAL_BATCH_SIZE[args.dataset_type]):
                image = sample['image'] # 1x3x400x400
                target = sample['mask'] # 1x400x400  int:0-11
                gt = target.data.cpu().numpy().astype(np.uint8)  #1x400x400 int:0-11
                input = image.data.cpu().numpy().astype(np.uint8)[0].transpose(1,2,0)  #1x400x400 int:0-11
                input_var = torch.autograd.Variable(image).float().cuda()
                if _out_type_1_:
                    # Compute output
                    output, _ = segmenter(input_var) #1x11x100x100 float
                    output = nn.Upsample(size=target.size()[1:], mode='bilinear',
                                         align_corners=False)(output) #1x11x400x400 float
                    # Compute IoU
                    image_name = val_loader.dataset.datalist[i][0].split('/')[1].split('.')[0]
                    output = output.data.cpu().numpy().argmax(axis=1).astype(np.uint8)  # 1x400x400 int:0-11
                    cv2.imwrite(os.path.join(validate_color_dir, "{}.png".format(image_name)), color_array[output[0]])
                    # cv2.imwrite(os.path.join(validate_gt_dir, "{}.png".format(image_name)), color_array[gt[0]])
                    cv2.imwrite(os.path.join(validate_output_dir, "{}.png".format(i)), output[0])
                    cv2.imwrite(os.path.join(validate_gt_dir, "{}.png".format(i)), gt[0])
                else:
                    if 1:
                        img_inp = torch.tensor(prepare_img(input).transpose(2, 0, 1)[None]).float().to(device)
                        segm = segmenter(img_inp)[0].squeeze().data.cpu().numpy().transpose((1, 2, 0))  # 47*63*21
                        # segm = cv2.resize(segm, tuple(target.size()[1:]), interpolation=cv2.INTER_CUBIC)  # 375*500*21
                        segm = cv2.resize(segm, target.size()[1:][::-1], interpolation=cv2.INTER_CUBIC)  # 375*500*21
                        segm = segm.argmax(axis=2).astype(np.uint8)
                        image_name = val_loader.dataset.datalist[i][0].split('/')[1].split('.')[0]
                        cv2.imwrite(os.path.join(validate_color_dir, "{}.png".format(image_name)), color_array[segm])
                        # cv2.imwrite(os.path.join(validate_gt_dir, "{}.png".format(image_name)), color_array[gt[0]])
                        cv2.imwrite(os.path.join(validate_output_dir,"{}.png".format(image_name)),segm)
                        cv2.imwrite(os.path.join(validate_gt_dir,"{}.png".format(image_name)),gt[0])

                    # segm = segmenter(input_var)[0].squeeze().data.cpu().numpy().transpose((1, 2, 0))  # 47*63*21
                    # # segm = cv2.resize(segm, target.size()[1:], interpolation=cv2.INTER_CUBIC)  # 375*500*21
                    # segm = cv2.resize(segm, (800,800), interpolation=cv2.INTER_CUBIC)  # 375*500*21
                    # segm = segm.argmax(axis=2).astype(np.uint8)
                    # gto = cv2.resize(gt[0], (800,800),interpolation=cv2.INTER_CUBIC)
                    # # gto = cv2.resize(gt[0],segm.shape,interpolation=cv2.INTER_CUBIC)
                    # cv2.imwrite(os.path.join(validate_output_dir, "{}.png".format(i)), segm)
                    # cv2.imwrite(os.path.join(validate_gt_dir, "{}.png".format(i)), gto)

        cal_f1_score(validate_gt_dir,validate_output_dir)

        # if i > 50:
            #     break
    else:
        task_miou = validate(segmenter,
                         val_loader,
                         1,
                         1, #[5,1]
                         num_classes=NUM_CLASSES[args.dataset_type][0],
                         print_every=args.print_every)

if __name__ == '__main__':
    main()
