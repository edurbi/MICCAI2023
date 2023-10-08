import argparse
import os
import time
import logging
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr


from model import Model
from Dataset_3D import Dataset_3D
from Stats import evaluate

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=4, type=int)
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--gpu', default="0", type=str)
path = os.path.dirname(__file__)
args = parser.parse_args()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.train_transforms = 'Compose([tio.transforms.CropOrPad((256,256,256)),T.ConvertImageDtype(torch.float32)])'
args.test_transforms = 'Compose([tio.transforms.CropOrPad((256,256,256)),T.ConvertImageDtype(torch.float32)])'
args.train_transforms_xr = 'Compose([T.Pad((600,600)),T.RandomCrop((2048,2048)),T.RandomRotation(10),PILToTensor(),T.ConvertImageDtype(torch.float32)])'
args.test_transforms_xr = 'Compose([T.Pad((600,600)),T.CenterCrop((2048,2048)),PILToTensor(),T.ConvertImageDtype(torch.float32)])'
ckpts = args.savepath

os.makedirs(ckpts, exist_ok=True)


def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')

    model = Model().cuda()


    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
    optimizer = torch.optim.Adam(train_params, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    train_stats = evaluate()
    val_stats = evaluate()
    logging.info(str(args))
    train_set = Dataset_3D(transforms_xr=args.test_transforms_xr, transforms=args.train_transforms, root=args.datapath,
                           type=0)
    val_set = Dataset_3D(transforms_xr=args.test_transforms_xr, transforms=args.test_transforms, root=args.datapath,
                         type=1)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=False)

    #lr_schedule = lr.CosineAnnealingLR(verbose=True, optimizer=optimizer, T_max=args.num_epochs * len(train_loader))
    lr_schedule = lr.CosineAnnealingWarmRestarts(verbose=True, optimizer=optimizer, T_0=10)
    start_epoch = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])
        log_file = open(os.path.join(ckpts, 'logging.txt'), 'a')
        val_stats.set_best_auc(checkpoint['best'])
        for g in optimizer.param_groups:
            g['lr'] = args.lr
    else:
        log_file = open(os.path.join(ckpts, 'logging.txt'), 'w+')
    start = time.time()
    msg = "Starting time: {:.1f}\n".format(start)
    log_file.write(msg)
    log_file.flush()
    iter_per_epoch = len(train_loader)
    train_iter = iter(train_loader)

    criterion = nn.CrossEntropyLoss()

    saved_train_auc = 0
    saved_val_auc = 0

    for epoch in range(start_epoch,args.num_epochs):
        logging.info('Train step')
        torch.set_grad_enabled(True)
        model.is_training = True
        model.train()
        b = time.time()
        old_data=None
        for i in range(iter_per_epoch):
            step = (i + 1) + epoch * iter_per_epoch
            try:
                data = next(train_iter)
                old_data=data
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, y, session, X_1, X_2 = data # age, gender, exists = data
            #age = age.cuda(non_blocking=True)
            #gender = gender.cuda(non_blocking=True)
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            X_1, X_2 = X_1.cuda(non_blocking=True), X_2.cuda(non_blocking=True)
            fuse_pred = model(x, len(y), X_1, X_2,False)
            loss = criterion(fuse_pred, y.to(torch.uint8))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_schedule.step(epoch + i / iter_per_epoch)
            train_stats.update(fuse_pred.cpu().detach(), y.cpu().to(torch.uint8), loss.item())
            msg = 'Train for Epoch {}/{}, Iter {}/{}, Loss {:.4f}, Acc {:.4f},Auc {:.4f},'.format((epoch + 1),
                                                                                                  args.num_epochs,
                                                                                                  (i + 1),
                                                                                                  iter_per_epoch,
                                                                                                  loss.item(),
                                                                                                  train_stats.total_acc(),
                                                                                                  train_stats.auc_score())
            print(msg)

        metrics = train_stats.confusion_matrix()
        logging.info('train time per epoch: {}'.format(time.time() - b))
        msg = "Train for Epoch {}/{}, Average Loss {:.4f}, Total Acc {:.4f},Total Auc {:.4f},True Positive {:d},False Positive {:d},False Negative {:d},True Negative{:d}\n".format(
            (epoch + 1), args.num_epochs, train_stats.avg_loss(), train_stats.total_acc(), train_stats.auc_score(),
            metrics[0][0], metrics[0][1], metrics[1][0], metrics[1][1])
        log_file.write(msg)
        log_file.flush()

        logging.info('Val step')
        with torch.no_grad():
            model.eval()
            model.is_training = False
            for i in range(len(val_loader)):
                try:
                    data = next(val_iter)
                except:
                    val_iter = iter(val_loader)
                    data = next(val_iter)
                x, y, session, X_1, X_2 = data#, age, gender, exists
                #age = age.cuda(non_blocking=True)
                #gender = gender.cuda(non_blocking=True)
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                X_1, X_2 = X_1.cuda(non_blocking=True), X_2.cuda(non_blocking=True)

                fuse_pred = model(x, y, X_1, X_2,False)

                loss = criterion(fuse_pred, y.to(torch.uint8))
                val_stats.update(fuse_pred.cpu().detach(), y.cpu().to(torch.uint8), loss.item())

                msg = 'Val Iter {}/{}, Acc {:.4f}, Auc {:.4f}'.format((i + 1), len(val_loader), val_stats.total_acc(),
                                                                      val_stats.auc_score())
                print(msg)

                logging.info(msg)
            metrics = val_stats.confusion_matrix()
            msg = "Val for Epoch {}/{}, Average Loss {:.4f}," \
                  " Total Acc {:.4f},Total Auc {:.4f},True Positive {:d},False Positive {:d},False Negative {:d},True Negative{:d}\n".format(
                (epoch + 1), args.num_epochs, val_stats.avg_loss(), val_stats.total_acc(), val_stats.auc_score(),
                metrics[0][0], metrics[0][1], metrics[1][0], metrics[1][1])
            print(msg)
            log_file.write(msg)
            log_file.flush()

        file_name = os.path.join(ckpts, 'model_last.pth')
        file_name2 = os.path.join(ckpts, 'secondary_model_last.pth')
        compare_auc = val_stats.auc_score()
        compare_auc_train = train_stats.auc_score()
        train_stats.new_epoch()
        val_stats.new_epoch()
        index, auc, _ = val_stats.find_best_auc()

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'model': model,
            'best': auc
        },

            os.path.join(ckpts, 'last.pth'))

        index_t, auc_t, _ = train_stats.find_best_auc()
        if index == epoch:
            saved_train_auc = auc_t
            msg = "Saved with auc:{:.4f} at the iteration\n".format(auc)
            log_file.write(msg)
            log_file.flush()
            print("Saved with auc:" + str(auc) + " at the iteration " + str(epoch))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'model': model,
                'best': auc
            },
                file_name)
        elif (auc - compare_auc) <= (auc * 0.03) and saved_val_auc <= compare_auc and (
                compare_auc + compare_auc_train) > (saved_val_auc + saved_train_auc):
            saved_train_auc = compare_auc_train
            saved_val_auc = compare_auc
            print("Saved second with val auc:" + str(saved_val_auc) + " train auc:" + str(
                saved_train_auc) + "val auc:" + str(saved_val_auc) + " at the iteration " + str(epoch))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'best': auc,
                'model': model
            },
                file_name2)

    index, auc, _ = val_stats.find_best_auc()
    msg = "Best Auc {:.4f} at the epoch{:.4f}\n".format(auc, index + 1)
    log_file.write(msg)
    log_file.flush()
    msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
    logging.info(msg)


if __name__ == '__main__':
    main()



