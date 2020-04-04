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


os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
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
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
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

    # Generate teacher if any
    # if args.do_kd:
    # 	from kd.rf_lw.model_lw_v2 import rf_lw152 as kd_model
    # 	global kd_net, kd_crit
    # 	kd_crit = nn.MSELoss().cuda()
    # 	kd_net = kd_model(
    # 		pretrained=True, num_classes=NUM_CLASSES[args.dataset_type][0]).cuda().eval()
    # 	logger.info(" Loaded teacher, #TOTAL PARAMS={:3.2f}M".format(
    # 		compute_params(kd_net)[0] / 1e6))

    # Generate controller / RL-agent
    # agent = create_agent(
    # 	args.op_size,
    # 	args.hidden_size,
    # 	args.num_lstm_layers,
    # 	args.num_cells,
    # 	args.num_branches,
    # 	args.lr_ctrl,
    # 	args.bl_dec,
    # 	args.agent_ctrl,
    # 	len(encoder.out_sizes))
    # logger.info(" Loaded Controller, #TOTAL PARAMS={:3.2f}M".format(
    # 	compute_params(agent.controller)[0] / 1e6))

    def create_segmenter(encoder):
        with torch.no_grad():
#			decoder_config, entropy, log_prob = agent.controller.sample()
            # stub the decoder arch
            # decoder_config = [[0, [0, 0, 5, 6], [4, 3, 5, 5], [2, 7, 2, 5]], [[3, 3], [2, 3], [4, 0]]]
            # decoder_config = [[5, [0, 0, 5, 1], [4, 0, 8, 7], [6, 3, 3, 2]], [[1, 1], [3, 3], [2, 0]]]
            # decoder_config = [[5, [0, 0, 3, 10], [3, 3, 7, 7], [7, 4, 7, 1]], [[0, 0], [2, 0], [0, 1]]] #0.7035
            # [[1, [1, 1, 5, 9], [2, 1, 9, 1], [3, 6, 1, 9]], [[1, 3], [1, 0], [4, 2]]] #0.7040 (reward)
            # [[6, [0, 1, 2, 5], [0, 2, 5, 3], [6, 5, 1, 10]], [[1, 3], [0, 3], [3, 4]]]  #0.7108 all cls
            # decoder_config = [[1, [0, 0, 6, 0], [0, 3, 8, 3], [3, 0, 6, 3]], [[1, 2], [2, 4], [0, 4]]] #0.7137 all cls
            # decoder_config = [[10, [1, 0, 8, 10], [0, 1, 3, 2], [7, 1, 4, 3]], [[3, 0], [3, 4], [3, 2]]] #0.095 worst all cls
            # [[10, [1, 1, 5, 2], [3, 0, 3, 4], [6, 7, 5, 9]], [[0, 0], [4, 3], [3, 1]]] #0.1293 all cls
            # decoder_config =   [[1, [1, 1, 5, 5], [1, 0, 2, 7], [1, 4, 7, 5]], [[2, 1], [3, 0], [3, 2]]] # 0.7601  new select
            decoder_config = [[3, [1, 1, 5, 0], [0, 4, 1, 9], [4, 3, 2, 0]], [[3, 3], [2, 1], [2, 0], [1,4]]]  #0.7564
            # decoder_config = [[5, [1, 0, 3, 5], [1, 0, 10, 10], [6, 6, 0, 10]], [[1, 0], [4, 2], [3, 2]]] # 0.7816 reward
            # decoder_config = [[5, [1, 0, 3, 5], [1, 0, 10, 10], [6, 6, 0, 10]], [[1, 0], [4, 2], [3, 2],[0,2],[1,4]]] # 0.7816 reward
            # decoder_config = [[5, [1, 0, 3, 5], [1, 0, 10, 10], [6, 6, 0, 10]], [[1, 0], [4, 2], [3, 2],[0,2],[1,4],[0,3]]] # 0.7816 reward
            # decoder_config = [[1, [0, 0, 10, 9], [0, 1, 2, 7], [2, 0, 0, 9]], [[2, 0], [3, 2], [2, 4]]] #0.9636 EG1800
            #decoder_config = [[1, [1, 0, 3, 9], [2, 3, 4, 9], [2, 1, 1, 1]], [[1, 3], [2, 0], [0, 3]]]  #0.9636 EG1800
            #decoder_config = [[2, [1, 0, 10, 8], [2, 3, 1, 8], [2, 1, 2, 2]], [[3, 1], [2, 4], [5, 5]]]
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
        return segmenter, decoder_config#, entropy, log_prob

    # Sample first configuration
    segmenter, decoder_config, = create_segmenter(encoder)
    del encoder

    # finetune_ckpt_path = './ckpt/_train_celebA-binary_20200118T1715/segmenter_checkpoint.pth.tar'
    # finetune_ckpt_path = './ckpt/_train_EG1800_20200217T1922/segmenter_checkpoint.pth.tar'
    #finetune_ckpt_path = './ckpt/_train_EG1800_20200218T1319/segmenter_checkpoint.pth.tar'
    #finetune_ckpt_path = './ckpt/_train_EG1800_20200218T2034/segmenter_checkpoint.pth.tar'
    # finetune_ckpt_path = './ckpt/_train_helen_20200223T1724/segmenter_checkpoint.pth.tar'
    # finetune_ckpt_path = './ckpt/_train_helen_20200226T2358/segmenter_checkpoint_0.36.pth.tar'
    # finetune_ckpt_path = './ckpt/_train_helen_20200227T2046/segmenter_checkpoint_0.37.pth.tar'
    # finetune_ckpt_path = './ckpt/_train_helen_20200228T1957/segmenter_checkpoint_0.31.pth.tar'
    # finetune_ckpt_path = './ckpt/_train_helen_20200228T2101/segmenter_checkpoint_0.30.pth.tar'
    # finetune_ckpt_path = './ckpt/_train_helen_20200301T1545/segmenter_checkpoint_0.28.pth.tar'
    # finetune_ckpt_path = './ckpt/_train_helen_20200302T2201/segmenter_checkpoint_0.27.pth.tar'
    # finetune_ckpt_path = './ckpt/_train_helen_nohair_20200303T1416/segmenter_checkpoint_0.19.pth.tar'
    # finetune_ckpt_path = './ckpt/_train_celebA-face_20200225T1518/segmenter_checkpoint_0.20.pth.tar'
    # finetune_ckpt_path = './ckpt/_train_celebA-face_20200225T1901/segmenter_checkpoint_0.14.pth.tar'
    # finetune_ckpt_path = './ckpt/_train_celebA_20200304T2257/segmenter_checkpoint_0.22.pth.tar'
    finetune_ckpt_path ='./ckpt/_train_celebA_20200305T1751/segmenter_checkpoint_0.25.pth.tar'
    segmenter.load_state_dict(torch.load(finetune_ckpt_path))
    logger.info(" Loaded Encoder with #TOTAL PARAMS={:3.2f}M"
                .format(compute_params(segmenter)[0] / 1e6))

    # Create dataloaders
    train_loader, val_loader, do_search = create_loaders(args)

    # Initialise task performance measurers
    task_ps = [[TaskPerformer(maxval=0.01, delta=0.9)
                for _ in range(TRAIN_EPOCH_NUM[args.dataset_type][idx] // args.val_every[idx])]
               for idx, _ in enumerate(range(args.num_tasks))]

    # Restore from previous checkpoint if any
    # best_val, epoch_start = load_ckpt(args.ckpt_path,
    #                                   {'agent': agent})

    # Saver: keeping checkpoint with best validation score (a.k.a best reward)
    now = datetime.datetime.now()

    args.snapshot_dir = args.snapshot_dir+'_train_'+args.dataset_type+"_{:%Y%m%dT%H%M}".format(now)
    # saver = Saver(args=vars(args),
    #               ckpt_dir=args.snapshot_dir,
    #               best_val=best_val,
    #               condition=lambda x, y: x > y)
    seg_saver=seg_Saver(ckpt_dir=args.snapshot_dir)

    arch_writer = open('{}/genotypes.out'.format(args.snapshot_dir), 'w')
    arch_writer.write(
        'genotype: {}\n'
            .format(decoder_config))
    arch_writer.flush()

    # with open('{}/args.json'.format(args.snapshot_dir), 'w') as f:
    #     json.dump({k: v for k, v in args.items() if isinstance(v, (int, float, str))}, f,
    #               sort_keys=True, indent=4, ensure_ascii=False)

    logger.info(" Pre-computing data for task0")
    kd_net = None# stub the kd
    # Xy_train = populate_task0(
    #     segmenter, train_loader, kd_net, N_TASK0[args.dataset_type], args.do_kd)
    # if args.do_kd:
    #     del kd_net

    logger.info(" Training Process Starts")
    for task_idx in range(args.num_tasks):#0,1
        # if stop:
        # 	break
        torch.cuda.empty_cache()
        # Change dataloader
        train_loader.batch_sampler.batch_size = BATCH_SIZE[args.dataset_type][task_idx]
        # for loader in [train_loader, val_loader]:
        #     try:
        #         loader.dataset.set_config(crop_size=args.crop_size[task_idx],
        #                                   shorter_side=args.shorter_side[task_idx])
        #     except AttributeError:
        #         # for subset
        #         loader.dataset.dataset.set_config(
        #             crop_size=args.crop_size[task_idx],
        #             shorter_side=args.shorter_side[task_idx])

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
        # avg_param = init_polyak(
        #     args.do_polyak, segmenter.module.decoder if task_idx == 0 else segmenter)
        kd_crit = None #stub the kd
        for epoch_segm in range(TRAIN_EPOCH_NUM[args.dataset_type][task_idx]): #[5,1] [20,8]
            if task_idx == 0:
                '''
                train_task0(Xy_train, #train the decoder once
                            segmenter,
                            optim_dec,
                            epoch_segm,#[5,1]
                            segm_crit,
                            kd_crit,
                            BATCH_SIZE[args.dataset_type][0],
                            args.freeze_bn[0],
                            args.do_kd,
                            args.kd_coeff,
                            args.dec_grad_clip,
                            args.do_polyak,
                            avg_param=avg_param,
                            polyak_decay=0.9,
                            aux_weight=args.dec_aux_weight)
                '''
                pass
            else:
                final_loss = train_segmenter(segmenter,  #train the segmenter end to end onece
                                train_loader,
                                optim_enc,
                                optim_dec,
                                epoch_segm, #[5,1]
                                segm_crit,
                                args.freeze_bn[1],
                                args.enc_grad_clip,
                                args.dec_grad_clip,
                                args.do_polyak,
                                args.print_every,
                                aux_weight=args.dec_aux_weight,
                                # avg_param=avg_param,
                                polyak_decay=0.99)
            # apply_polyak(args.do_polyak,
            #              segmenter.module.decoder if task_idx == 0 else segmenter,
            #              avg_param)
            #if (epoch_segm + 1) % (args.val_every[task_idx]) == 0:
            if False:
                logger.info(
                    " Validating Segmenter, Epoch {}, Task {}"
                        .format(str(9876), str(task_idx)))
                task_miou = validate(segmenter,
                                     val_loader,
                                     9876,
                                     epoch_segm, #[5,1]
                                     num_classes=NUM_CLASSES[args.dataset_type][task_idx],
                                     print_every=args.print_every)
                # Verifying if we are continuing training this architecture.
                c_task_ps = task_ps[task_idx][(epoch_segm + 1) // args.val_every[task_idx] - 1]
                if c_task_ps.step(task_miou):
                    continue
                else:
                    logger.info(" Interrupting")
                    stop = True
                    break
                #reward = task_miou  # will be used in train_agent process
            # save the segmenter params with the best value
            # seg_saver.save(final_loss, segmenter.state_dict(), logger) #stub to 1
    seg_saver.save(final_loss, segmenter.state_dict(), logger) #stub to 1


if __name__ == '__main__':
    main()
