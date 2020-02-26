## DEFAULT CONFIGURATION USED IN OUR EXPERIMENTS ON 2 GPUs

import numpy as np

# DATASET PARAMETERS
dataset_dirs = {
    'face_seg':
        {
            'TRAIN_DIR':'../data/datasets/face_seg_dataset/',
            'VAL_DIR':'../data/datasets/face_seg_dataset/',
            'TRAIN_LIST':'../data/datasets/face_seg_dataset/train.lst',
            'VAL_LIST' : '../data/datasets/face_seg_dataset/train.lst'  # meta learning
        },
    'celebA':
        {
            'TRAIN_DIR': '../data/datasets/portrait_parsing/',
            'VAL_DIR' : '../data/datasets/portrait_parsing/',
            #TRAIN_LIST = '../data/datasets/celebA/train_mini.lst'
            #VAL_LIST = '../data/datasets/celebA/train_mini.lst'  # meta learning
            'TRAIN_LIST' : '../data/datasets/portrait_parsing/train.lst',
            'VAL_LIST' : '../data/datasets/portrait_parsing/train.lst'  # meta learning
        },
    'EG1800':
        {
            'TRAIN_DIR': '../data/datasets/portrait_seg/EG1800/',
            'VAL_DIR': '../data/datasets/portrait_seg/EG1800/',
            'TRAIN_LIST': '../data/datasets/portrait_seg/EG1800/train.lst',
            'VAL_LIST': '../data/datasets/portrait_seg/EG1800/train.lst'  # meta learning
        },
    'celebA-binary':
        {
            'TRAIN_DIR': '../data/datasets/portrait_parsing/',
            'VAL_DIR': '../data/datasets/portrait_parsing/',
            # TRAIN_LIST = '../data/datasets/celebA/train_mini.lst'
            # VAL_LIST = '../data/datasets/celebA/train_mini.lst'  # meta learning
            'TRAIN_LIST': '../data/datasets/portrait_parsing/train_binary.lst',
            'VAL_LIST': '../data/datasets/portrait_parsing/train_binary.lst'  # meta learning
        },
    'helen':
        {
            'TRAIN_DIR': '../data/datasets/helen/',
            'VAL_DIR': '../data/datasets/helen/',
            # TRAIN_LIST = '../data/datasets/celebA/train_mini.lst'
            # VAL_LIST = '../data/datasets/celebA/train_mini.lst'  # meta learning
            # 'TRAIN_LIST': '../data/datasets/helen/train.lst',
            'TRAIN_LIST': '../data/datasets/helen/train_single.lst',
            # 'VAL_LIST': '../data/datasets/helen/val.lst'  # meta learning
            'VAL_LIST': '../data/datasets/helen/val_single.lst',  # meta learning
            # 'VAL_LIST': '../data/datasets/helen/train.lst',  # meta learning
        },
    'celebA-face':
        {
            'TRAIN_DIR': '../data/datasets/celebA/',
            'VAL_DIR': '../data/datasets/celebA/',
            # TRAIN_LIST = '../data/datasets/celebA/train_mini.lst'
            # VAL_LIST = '../data/datasets/celebA/train_mini.lst'  # meta learning
            'TRAIN_LIST': '../data/datasets/celebA/train_face_seg.lst',
            # 'VAL_LIST': '../data/datasets/helen/val.lst'  # meta learning
            'VAL_LIST': '../data/datasets/celebA/train_face_seg.lst'  # meta learning
        },

}


META_TRAIN_PRCT = 83
N_TASK0 = {'face_seg':1000,'celebA':1000,'EG1800':1000,'celebA-binary':1000,'helen':1000}
SHORTER_SIDE = [300, 400]
CROP_SIZE = [256, 350]
NORMALISE_PARAMS = [1./255, # SCALE
                    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)), # MEAN
                    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))] # STD
BATCH_SIZE ={'celebA':[64, 128],'EG1800':[16,1],'celebA-binary':[64,16],'helen':[16,16],'celebA-face':[16,32]}
NUM_WORKERS = 32
TRAIN_EPOCH_NUM = {'celebA':[40,10],'EG1800':[0,20],'celebA-binary':[0,6],'helen':[0,50],'celebA-face':[0,10]}

NUM_CLASSES = {'face_seg':[11,11],'celebA':[19,19],'EG1800':[2,2],'celebA-binary':[2,2], 'helen':[11,11],'celebA-face':[11,11]}
LOW_SCALE = 0.7
HIGH_SCALE = 1.4
VAL_SHORTER_SIDE = 512
VAL_CROP_SIZE = 512
VAL_BATCH_SIZE = {'face_seg':64,'celebA':64, 'EG1800':4,'celebA-binary':16,'helen':1,'celebA-face':1}

# ENCODER PARAMETERS
ENC_GRAD_CLIP = 3.

# DECODER PARAMETERS
DEC_GRAD_CLIP = 3.
DEC_AUX_WEIGHT = 0.15 # to disable aux, set to -1

# GENERAL
FREEZE_BN = [False, False]
NUM_EPOCHS = 100 #400 #20000
NUM_SEGM_EPOCHS = [40, 10] #[20, 8]#task 0(only decoder)for 20,task 1(end to end)for 8
PRINT_EVERY = 200
RANDOM_SEED = 9314
SNAPSHOT_DIR = './ckpt/'
#CKPT_PATH = './ckpt/20200107T2001/checkpoint.pth.tar'
#CKPT_PATH='./ckpt/2020testtttt'
VAL_EVERY = [10, 5] #10,4  # how often to record validation scores ; task0 valid for every 5 eopch , task1 valid for every 1 epoch
SUMMARY_DIR = './tb_logs/'

# OPTIMISERS' PARAMETERS
LR_ENC = [4e-3, 4e-3]
LR_DEC = [1e-2, 1e-2]

# LR_ENC = [1e-4, 1e-4]# finetune
# LR_DEC = [3e-4, 3e-4]#finetune

LR_CTRL = 1e-4
MOM_ENC = [0.9] * 3
MOM_DEC = [0.9] * 3
MOM_CTRL = 0.9
WD_ENC = [1e-5] * 3
WD_DEC = [0] * 3
WD_CTRL = 1e-4
OPTIM_DEC = 'adam'
OPTIM_ENC = 'sgd'
AGENT_CTRL = 'ppo'
DO_KD = False#True
KD_COEFF = 0.3
DO_POLYAK = True

# CONTROLLER
BL_DEC = 0.95
OP_SIZE = 11
AGG_CELL_SIZE = 48
NUM_CELLS = 3
NUM_BRANCHES = 4
AUX_CELL = True
SEP_REPEATS = 1
