import argparse
import os
import time
import logging
import random
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
import skimage.transform as skTrans
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr
from torch.profiler import profile, record_function, ProfilerActivity

import torchio as tio


from model import Model
from utils import Parser
from utils.parser import setup
from Stats import AUCRecorder
from Dataset_3D import Dataset_3D
from predict import AverageMeter, test_softmax
from Stats import evaluate

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=4, type=int, help='Batch size')
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--seed', default=1024, type=int)
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
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')

    ##########setting models
    model = Model()
    #model= torch.compile(model)
    #print (model)
    model = torch.nn.DataParallel(model).cuda()

    ##########Setting learning schedule and optimizer
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    ##########Setting data
    train_stats=evaluate()
    val_stats=evaluate()
    logging.info(str(args))
    train_set = Dataset_3D(transforms_xr=args.test_transforms_xr,transforms=args.train_transforms, root=args.datapath,type=0)
    val_set = Dataset_3D(transforms_xr=args.test_transforms_xr,transforms=args.test_transforms,root=args.datapath,type=2)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True)
    val_loader = MultiEpochsDataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2)

    lr_schedule = lr.CosineAnnealingLR(verbose=True,optimizer=optimizer,T_max=args.num_epochs*len(train_loader))
    #lr_schedule = lr.ReduceLROnPlateau(verbose=True,optimizer=optimizer)

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
    for g in optimizer.param_groups:
        g['lr'] = 1e-7
    logging.info('#############training############')
    msg = "Starting time: {:.1f}\n".format(start)
    log_file.write(msg)
    log_file.flush()
    iter_per_epoch = len(train_loader)
    train_iter = iter(train_loader)

    criterion=nn.CrossEntropyLoss()

    saved_train_auc = 0
    saved_val_auc = 0


    for epoch in range(args.num_epochs):
        torch.set_grad_enabled(True)
        model.module.is_training = True
        model.train()
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i + 1) + epoch * iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, y, name,session,X_1,X_2,age,gender,exists= data
            age= age.cuda(non_blocking=True)
            gender= gender.cuda(non_blocking=True)
            x = x.cuda(non_blocking=True).requires_grad_()
            y = y.cuda(non_blocking=True)
            X_1, X_2 = X_1.cuda(non_blocking=True),X_2.cuda(non_blocking=True)
            fuse_pred = model(x, len(y),X_1,X_2,age,gender,True,exists)
            output_idx = fuse_pred[0].argmax()
            ##loss
            loss = criterion(fuse_pred, y.to(torch.uint8))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_schedule.step()

            if output_idx==0:
                output_max = fuse_pred[0, 0]

                #output_max.backward()

                saliency, _ = torch.max(x.grad.data.abs(), dim=1)
                ct = x.cpu().detach().numpy()[0,0].transpose(2,1,0)
                saliency = saliency.cpu().detach().numpy()[0].transpose(2,1,0)

                for p in range(40,160,10):
                    fig, ax = plt.subplots(1, 2)
                    ax[0].imshow(ct[p], cmap='gray')
                    ax[0].axis('off')
                    ax[1].imshow(saliency[p], cmap='hot')
                    ax[1].axis('off')
                    plt.tight_layout()
                    fig.suptitle('The Image and Its Saliency Map')
                    plt.show()


            ###log
            train_stats.update(fuse_pred.cpu().detach(), y.cpu().to(torch.uint8),loss.item())
            writer.add_scalar('loss', loss.item(), global_step=step)
            msg = 'Train for Epoch {}/{}, Iter {}/{}, Loss {:.4f}, Acc {:.4f},Auc {:.4f},'.format((epoch + 1), args.num_epochs, (i + 1), iter_per_epoch,
                                                                  loss.item(),train_stats.total_acc(),train_stats.auc_score())
            logging.info(msg)

        metrics = train_stats.confusion_matrix()
        logging.info('train time per epoch: {}'.format(time.time() - b))
        msg = "Train for Epoch {}/{}, Average Loss {:.4f}, Total Acc {:.4f},Total Auc {:.4f},True Positive {:d},False Positive {:d},False Negative {:d},True Negative{:d}\n".format((epoch + 1), args.num_epochs,train_stats.avg_loss(),train_stats.total_acc(),train_stats.auc_score(), metrics[0][0],metrics[0][1],metrics[1][0],metrics[1][1])
        log_file.write(msg)
        log_file.flush()

        with torch.no_grad():
            model.eval()
            model.module.is_training = False
            for i in range(len(val_loader)):
                ###Data load
                try:
                    data = next(val_iter)
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


                loss = criterion(fuse_pred, y.to(torch.uint8))
                val_stats.update(fuse_pred.cpu().detach(), y.cpu().to(torch.uint8), loss.item())

                msg = 'Val Iter {}/{}, Acc {:.4f}, Auc {:.4f}'.format((i + 1), len(val_loader),val_stats.total_acc(),val_stats.auc_score())

                logging.info(msg)
            metrics = val_stats.confusion_matrix()
            msg = "Val for Epoch {}/{}, Average Loss {:.4f}," \
                  " Total Acc {:.4f},Total Auc {:.4f},True Positive {:d},False Positive {:d},False Negative {:d},True Negative{:d}\n".format((epoch + 1), args.num_epochs, val_stats.avg_loss(),val_stats.total_acc(),val_stats.auc_score(), metrics[0][0],metrics[0][1],metrics[1][0],metrics[1][1])
            log_file.write(msg)
            log_file.flush()


        ##########model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        file_name2 = os.path.join(ckpts, 'secondary_model_last.pth')
        compare_auc = val_stats.auc_score()
        compare_auc_train = train_stats.auc_score()
        train_stats.new_epoch()
        val_stats.new_epoch()

        index,auc,_= val_stats.find_best_auc()
        index_t, auc_t, _ = train_stats.find_best_auc()
        if index == epoch:
            saved_train_auc = auc_t
            msg = "Saved with auc:{:.4f} at the iteration\n".format(auc)
            log_file.write(msg)
            log_file.flush()
            print("Saved with auc:"+str(auc)+" at the iteration "+str(epoch))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'model': model,
                'best':auc
            },
                file_name)
        elif (auc-compare_auc)<=(auc*0.03) and saved_val_auc<=compare_auc and (compare_auc+compare_auc_train)>(saved_val_auc+saved_train_auc):
            saved_train_auc= compare_auc_train
            saved_val_auc= compare_auc
            print("Saved second with val auc:" + str(saved_val_auc)+ " train auc:" + str(saved_train_auc) + "val auc:" + str(saved_val_auc) + " at the iteration " + str(epoch))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'best': auc,
                'model':model
            },
                file_name2)

    index, auc, _ = val_stats.find_best_auc()
    msg = "Best Auc {:.4f} at the epoch{:.4f}\n".format(auc,index+1)
    log_file.write(msg)
    log_file.flush()
    msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
    logging.info(msg)



if __name__ == '__main__':
    main()



