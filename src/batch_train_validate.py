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
from utils.solvers import create_optimisers
from PIL import  Image
import cv2
import matplotlib.pyplot as plt
import  shutil
from utils.helpers import prepare_img
from utils.f1_score import *

os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
logging.basicConfig(level=logging.INFO)
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="NAS Search")

    parser.add_argument("--dataset_type", type=str, default= 'celebA',#'helen', #'celebA-face',#'EG1800',#'celebA-binary',
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
    parser.add_argument("--snapshot-dir", type=str, default='./ckpt/batch_training',
                        help="Path to directory for storing checkpoints.")
    # parser.add_argument("--ckpt-path", type=str, default=CKPT_PATH,
    #                     help="Path to the checkpoint file.")
    parser.add_argument("--val-every", nargs='+', type=int, default=VAL_EVERY,
                        help="How often to validate current architecture.")
    parser.add_argument("--summary-dir", type=str, default=SUMMARY_DIR,
                        help="Summary directory.")

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

    return parser.parse_args()


class Segmenter(nn.Module):
    """Create Segmenter"""
    def __init__(self, encoder, decoder):
        super(Segmenter, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

decoder_config_arry = [
    # first iter
    [[2, [1, 0, 7, 7], [1, 2, 3, 2], [7, 7, 8, 8]], [[1, 0], [0, 2], [2, 5]]],#reward: 0.0038, Epoch: 516, params: 1995073, epoch_time: 2.7239, genotype:
    [[6, [0, 1, 2, 3], [4, 1, 3, 0], [5, 2, 6, 1]], [[1, 3], [2, 0], [3, 5]]],#reward: 0.0019, Epoch: 151, params: 2311873, epoch_time: 3.0982, genotype:
    [[4, [1, 1, 2, 8], [3, 3, 3, 9], [1, 5, 6, 8]], [[2, 2], [2, 3], [3, 0]]],#reward: 0.1444, Epoch: 116, params: 2085793, epoch_time: 27.7913, genotype:
    [[4, [1, 1, 6, 9], [1, 1, 8, 7], [7, 1, 6, 8]], [[1, 2], [2, 3], [2, 4]]],#reward: 0.1455, Epoch: 150, params: 2186881, epoch_time: 6.8284, genotype:
    [[3, [1, 1, 4, 2], [0, 2, 2, 1], [4, 4, 0, 8]], [[1, 3], [2, 0], [3, 3]]],#reward: 0.1630, Epoch: 670, params: 2095585, epoch_time: 2.8670, genotype:
    [[3, [1, 1, 6, 1], [0, 2, 2, 9], [3, 0, 3, 9]], [[1, 1], [1, 2], [2, 2]]],#reward: 0.2595, Epoch: 3650, params: 2174785, epoch_time: 3.1443, genotype:
    [[4, [0, 0, 7, 9], [0, 2, 5, 1], [4, 6, 0, 7]], [[1, 1], [1, 1], [5, 3]]],#reward: 0.2978, Epoch: 2932, params: 2175073, epoch_time: 3.6690, genotype:
    [[1, [0, 1, 3, 8], [0, 3, 6, 10], [2, 0, 0, 9]], [[1, 3], [3, 1], [3, 1]]],#reward: 0.2520, Epoch: 3273, params: 2172193, epoch_time: 3.1918, genotype:
    [[9, [1, 0, 1, 1], [3, 3, 0, 0], [7, 6, 4, 2]], [[3, 3], [3, 4], [5, 5]]],#reward: 0.3777, Epoch: 474, params: 2170177, epoch_time: 17.4807, genotype:
    [[7, [1, 0, 5, 4], [3, 3, 8, 9], [4, 1, 0, 2]], [[3, 2], [3, 2], [3, 1]]],#reward: 0.3561, Epoch: 823, params: 2073985, epoch_time: 13.7779, genotype:
    [[1, [1, 1, 1, 10], [0, 2, 5, 6], [7, 5, 10, 4]], [[1, 1], [3, 1], [5, 1]]],#reward: 0.3689, Epoch: 3162, params: 2376673, epoch_time: 5.5331, genotype:
    [[4, [0, 1, 7, 4], [1, 4, 6, 7], [4, 1, 7, 3]], [[2, 3], [3, 2], [5, 4]]],#reward: 0.4550, Epoch: 749, params: 2086369, epoch_time: 21.2817, genotype:
    [[9, [0, 0, 9, 6], [3, 1, 8, 3], [7, 5, 2, 6]], [[2, 2], [2, 2], [5, 4]]],#reward: 0.4849, Epoch: 167, params: 2170177, epoch_time: 19.0512, genotype:
    [[3, [1, 1, 6, 0], [4, 1, 1, 7], [4, 3, 8, 4]], [[3, 2], [3, 4], [3, 2]]],#reward: 0.4993, Epoch: 146, params: 2201281, epoch_time: 19.2926, genotype:
    [[9, [1, 1, 4, 9], [1, 3, 0, 1], [4, 6, 7, 9]], [[2, 2], [2, 2], [3, 2]]],#reward: 0.5047, Epoch: 822, params: 2035393, epoch_time: 18.0973, genotype:
    [[5, [1, 1, 6, 8], [0, 4, 8, 2], [4, 6, 1, 8]], [[2, 3], [2, 2], [2, 2]]],#reward: 0.5011, Epoch: 176, params: 2321377, epoch_time: 19.8615, genotype:
    [[5, [1, 0, 2, 3], [2, 1, 4, 0], [2, 6, 10, 10]], [[1, 3], [4, 2], [5, 5]]],#reward: 0.5912, Epoch: 224, params: 2052385, epoch_time: 24.6115, genotype:
     [[4, [1, 0, 9, 9], [1, 1, 9, 1], [7, 5, 9, 9]], [[3, 1], [2, 2], [3, 2]]],#reward: 0.6391, Epoch: 3334, params: 2004001, epoch_time: 11.8697, genotype:
     [[6, [0, 1, 7, 7], [0, 3, 0, 7], [4, 5, 8, 4]], [[2, 3], [1, 3], [5, 3]]],#reward: 0.6763, Epoch: 285, params: 2088673, epoch_time: 22.9851, genotype:
     [[10, [1, 0, 0, 0], [0, 0, 9, 9], [4, 1, 9, 9]], [[2, 1], [3, 4], [5, 2]]],#reward: 0.6839, Epoch: 3627, params: 1888801, epoch_time: 11.0803, genotype:
     [[8, [1, 0, 6, 6], [0, 3, 10, 4], [3, 5, 9, 6]], [[1, 3], [3, 1], [4, 3]]],#reward: 0.6905, Epoch: 3582, params: 2273281, epoch_time: 15.4546, genotype:
     [[2, [0, 1, 8, 10], [4, 0, 2, 0], [7, 5, 8, 3]], [[1, 1], [1, 2], [2, 1]]],#reward: 0.7208, Epoch: 328, params: 1977793, epoch_time: 11.9111, genotype:
     [[6, [1, 1, 5, 5], [0, 2, 3, 0], [5, 6, 0, 2]], [[3, 1], [2, 1], [2, 2]]],#reward: 0.7400, Epoch: 346, params: 2306977, epoch_time: 14.4050, genotype:
     [[1, [1, 0, 1, 3], [0, 2, 10, 9], [4, 0, 9, 9]], [[2, 1], [2, 3], [5, 3]]],#reward: 0.7488, Epoch: 3576, params: 2133889, epoch_time: 11.1181, genotype
     [[1, [0, 0, 5, 1], [3, 2, 1, 10], [3, 5, 10, 9]], [[2, 3], [1, 2], [2, 2]]], #reward: 0.7613, Epoch: 2011, params: 2364577, epoch_time: 10.9261, genotype
    # second iter
    [[5, [0, 1, 7, 9], [3, 1, 7, 9], [4, 6, 1, 4]], [[0, 3], [4, 2], [4, 5]]],  #reward: 0.0659, Epoch: 466, params: 2158369, epoch_time: 2.9258, genotype:
    [[6, [0, 0, 8, 6], [2, 1, 0, 0], [4, 5, 7, 9]], [[3, 0], [4, 2], [3, 1]]] , #reward: 0.0664, Epoch: 456, params: 2179681, epoch_time: 3.0532, genotype:
    [[9, [1, 0, 9, 8], [3, 4, 7, 0], [4, 4, 1, 10]], [[1, 0], [4, 4], [1, 4]]] ,#reward: 0.0349, Epoch: 478, params: 2040289, epoch_time: 2.5031, genotype:
    [[5, [1, 1, 3, 7], [1, 4, 5, 1], [1, 6, 7, 8]], [[1, 1], [0, 1], [1, 1]]]  ,#reward: 0.0566, Epoch: 383, params: 2316769, epoch_time: 2.9454, genotype:
    [[10, [1, 1, 5, 8], [1, 1, 5, 5], [4, 2, 9, 2]], [[2, 1], [2, 0], [4, 0]]] ,#reward: 0.0566, Epoch: 142, params: 2275873, epoch_time: 3.3682, genotype:
     [[8, [0, 0, 0, 2], [0, 2, 10, 1], [4, 0, 5, 8]], [[1, 1], [3, 1], [2, 3]]],#reward: 0.2544, Epoch: 3372, params: 2189185, epoch_time: 3.1039, genotype:
     [[3, [1, 0, 3, 1], [0, 2, 10, 9], [3, 1, 0, 2]], [[1, 1], [1, 2], [2, 2]]],#reward: 0.2593, Epoch: 3511, params: 2064193, epoch_time: 2.8533, genotype:
     [[3, [1, 0, 10, 0], [0, 3, 3, 3], [3, 6, 6, 6]], [[2, 1], [3, 1], [2, 1]]],#reward: 0.2803, Epoch: 2861, params: 2193793, epoch_time: 3.5468, genotype:
     [[1, [1, 0, 9, 8], [0, 3, 1, 3], [3, 0, 5, 0]], [[1, 1], [1, 2], [2, 2]]], #reward: 0.3012, Epoch: 3259, params: 2297185, epoch_time: 3.3084, genotype:
     [[5, [1, 0, 9, 9], [3, 4, 3, 2], [4, 1, 8, 6]], [[2, 3], [3, 2], [3, 2]]], #reward: 0.5138, Epoch: 391, params: 2174785, epoch_time: 18.7520, genotype:
     [[1, [1, 1, 9, 10], [0, 4, 2, 3], [3, 0, 3, 5]], [[2, 3], [2, 2], [2, 2]]],#reward: 0.5414, Epoch: 982, params: 2174785, epoch_time: 18.6489, genotype:
]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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


    # Create dataloaders
    train_loader, val_loader, do_search = create_loaders(args)

    def create_segmenter(encoder, decoder_config):
        with torch.no_grad():
            decoder = Decoder(inp_sizes=encoder.out_sizes,
                                          num_classes=NUM_CLASSES[args.dataset_type][0],
                                          config=decoder_config,
                                          agg_size=48,   #args.agg_cell_size, what's the fxxk
                                          aux_cell=True,  #args.aux_cell,
                                          repeats=1)#args.sep_repeats)

        # Fuse encoder and decoder
        segmenter = nn.DataParallel(Segmenter(encoder, decoder)).cuda()
        logger.info(" Created Segmenter, #PARAMS (Total, No AUX)={}".format(
            compute_params(segmenter)
        ))
        return segmenter#, entropy, log_prob

    for decoder_config in decoder_config_arry:
        # Initialise encoder
        encoder = create_encoder()
        logger.info(" Loaded Encoder with #TOTAL PARAMS={:3.2f}M"
                    .format(compute_params(encoder)[0] / 1e6))
        # Sample first configuration
        segmenter = create_segmenter(encoder, decoder_config)
        del encoder

        logger.info(" Loaded Encoder with #TOTAL PARAMS={:3.2f}M"
                    .format(compute_params(segmenter)[0] / 1e6))

        # Saver: keeping checkpoint with best validation score (a.k.a best reward)
        now = datetime.datetime.now()

        snapshot_dir = args.snapshot_dir+'_train_'+args.dataset_type+"_{:%Y%m%dT%H%M}".format(now)
        seg_saver=seg_Saver(ckpt_dir=snapshot_dir)

        arch_writer = open('{}/genotypes.out'.format(snapshot_dir), 'w')
        arch_writer.write(
            'genotype: {}\n'
                .format(decoder_config))
        arch_writer.flush()

        logger.info(" Pre-computing data for task0")
        kd_net = None# stub the kd

        logger.info(" Training Process Starts")
        for task_idx in range(args.num_tasks):#0,1
            if task_idx == 0:
                continue
            torch.cuda.empty_cache()
            # Change dataloader
            train_loader.batch_sampler.batch_size = BATCH_SIZE[args.dataset_type][task_idx]

            logger.info(" Training Task {}".format(str(task_idx)))
            # Optimisers
            optim_enc, optim_dec = create_optimisers(
                args.optim_enc,
                args.optim_dec,
                args.lr_enc[task_idx],
                args.lr_dec[task_idx],
                args.mom_enc[task_idx],
                args.mom_dec[task_idx],
                args.wd_enc[task_idx],
                args.wd_dec[task_idx],
                segmenter.module.encoder.parameters(),
                segmenter.module.decoder.parameters())
            kd_crit = None #stub the kd
            for epoch_segm in range(TRAIN_EPOCH_NUM[args.dataset_type][task_idx]):  # [5,1] [20,8]
                final_loss = train_segmenter(segmenter,  #train the segmenter end to end onece
                                train_loader,
                                optim_enc,
                                optim_dec,
                                epoch_segm,
                                segm_crit,
                                args.freeze_bn[1],
                                args.enc_grad_clip,
                                args.dec_grad_clip,
                                args.do_polyak,
                                args.print_every,
                                aux_weight=args.dec_aux_weight,
                                # avg_param=avg_param,
                                polyak_decay=0.99)
        seg_saver.save(final_loss, segmenter.state_dict(), logger) #stub to 1
        # validat
        segmenter.eval()
        data_file=dataset_dirs[args.dataset_type]['VAL_LIST']
        data_dir=dataset_dirs[args.dataset_type]['VAL_DIR']
        with open(data_file, 'rb') as f:
            datalist = f.readlines()
        try:
            datalist = [
                (k, v) for k, v, _ in \
                map(lambda x: x.decode('utf-8').strip('\n').split('\t'), datalist)]
        except ValueError:  # Adhoc for test.
            datalist = [
                (k, k) for k in map(lambda x: x.decode('utf-8').strip('\n'), datalist)]
        imgs_all = [os.path.join(data_dir, datalist[i][0]) for i in range(0, len(datalist))]
        msks_all = [os.path.join(data_dir, datalist[i][1]) for i in range(0, len(datalist))]
        validate_output_dir = os.path.join(dataset_dirs[args.dataset_type]['VAL_DIR'], 'validate_output')
        validate_gt_dir = os.path.join(dataset_dirs[args.dataset_type]['VAL_DIR'], 'validate_gt')
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
        # validate_color_dir = os.path.join(dataset_dirs[args.dataset_type]['VAL_DIR'], 'validate_output_color')
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
            # cv2.imwrite(os.path.join(validate_color_dir, "{}.png".format(image_name)), color_array[segm])
            # cv2.imwrite(os.path.join(validate_gt_dir, "{}.png".format(image_name)), color_array[msk])
            cv2.imwrite(os.path.join(validate_output_dir, "{}.png".format(image_name)), segm)
            cv2.imwrite(os.path.join(validate_gt_dir, "{}.png".format(image_name)), msk)

        if args.dataset_type == 'celebA':
            cal_f1_score_celebA(validate_gt_dir,validate_output_dir, arch_writer) # temp comment

if __name__ == '__main__':
    main()
