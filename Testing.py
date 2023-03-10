import argparse
import os
import time
import logging
import random
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn
import sklearn
import skimage.transform as skTrans
from collections import OrderedDict
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr
import scikitplot as skplt
from torch.profiler import profile, record_function, ProfilerActivity

import torchio as tio


from model import Model
from data.transforms import *
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup
from Stats import AUCRecorder
from Dataset_3D import Dataset_3D
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader
from predict import AverageMeter, test_softmax
from Stats import evaluate



parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=6, type=int, help='Batch size')
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=40, type=int)
path = os.path.dirname(__file__)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

args = parser.parse_args()
setup(args,'training')
args.train_transforms = 'Compose([tio.transforms.CropOrPad((256,256,256)),T.ConvertImageDtype(torch.float32)])'
args.test_transforms = 'Compose([tio.transforms.CropOrPad((256,256,256)),T.ConvertImageDtype(torch.float32)])'
args.train_transforms_xr ='Compose([T.Pad((600,600)),T.RandomCrop((2048,2048)),T.RandomRotation(10),PILToTensor(),T.ConvertImageDtype(torch.float32)])'
args.test_transforms_xr ='Compose([T.Pad((600,600)),T.CenterCrop((2048,2048)),PILToTensor(),T.ConvertImageDtype(torch.float32)])'
ckpts = args.savepath

os.makedirs(ckpts, exist_ok=True)
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

def main():
    ##########setting seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')

    ##########setting models
    model = Model()
    model = torch.nn.DataParallel(model).cuda()

    ##########Setting learning schedule and optimizer
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    ##########Setting data
    val_stats=evaluate()
    logging.info(str(args))
    val_set = Dataset_3D(transforms_xr=args.test_transforms_xr,transforms=args.test_transforms,root=args.datapath,type=2)
    val_loader = MultiEpochsDataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2)


    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])
        log_file = open(os.path.join(ckpts, 'logging.txt'), 'a')
        val_stats.set_best_auc(checkpoint['best'])
    else:
        log_file = open(os.path.join(ckpts, 'logging.txt'), 'w+')
    start = time.time()
    logging.info('#############training############')
    msg = "Starting time: {:.1f}\n".format(start)
    log_file.write(msg)
    log_file.flush()

    criterion=nn.CrossEntropyLoss()

    with torch.no_grad():
        res =None
        true=None
        model.eval()
        model.module.is_training = False
        for i in range(len(val_loader)):
            try:
                data = next(val_loader)
            except:
                val_iter = iter(val_loader)
                data = next(val_iter)
            x, y, name,session,X_1,X_2,age,gender,exists= data
            age = age.cuda(non_blocking=True)
            gender = gender.cuda(non_blocking=True)
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            X_1, X_2 = X_1.cuda(non_blocking=True),X_2.cuda(non_blocking=True)

            fuse_pred = model(x, y,X_1,X_2,age,gender,False,exists)
            if i ==0:
                res=np.array(fuse_pred.cpu().detach())
            else:
                res = np.concatenate((res,np.array(fuse_pred.cpu().detach())))
            if i==0:
                true = np.array(y.cpu().detach())
            else:
                true = np.concatenate((true,np.array(y.cpu().detach())))

            loss = criterion(fuse_pred, y.to(torch.uint8))
            val_stats.update(fuse_pred.cpu().detach(), y.cpu().to(torch.uint8), loss.item())

            msg = 'Val Iter {}/{}, Acc {:.4f}, Auc {:.4f}'.format((i + 1), len(val_loader),val_stats.total_acc(),val_stats.auc_score())

            logging.info(msg)
        metrics = val_stats.confusion_matrix()
        msg = "Val Average Loss {:.4f}," \
              " Total Acc {:.4f},Total Auc {:.4f},True Positive {:.4f},False Positive {:.4f},False Negative {:.4f},True Negative{:.4f}\n".format(val_stats.avg_loss(),val_stats.total_acc(),val_stats.auc_score(), metrics[0][0],metrics[0][1],metrics[1][0],metrics[1][1])
        print(msg)
        skplt.metrics.plot_roc(true, res)
        plt.show()
        df_cm = pd.DataFrame(metrics, index=["Positive","Negative"],columns=["Positive","Negative"])
        plt.figure(figsize=(2,2))
        seaborn.set(font_scale=1.4)
        seaborn.heatmap(df_cm, annot=True)
        plt.show()




if __name__ == '__main__':
    main()




