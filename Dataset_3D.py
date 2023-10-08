import os
import torch
import time
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as f
from torchvision.transforms import Resize,RandomAffine
from torchvision.transforms import Compose,PILToTensor
import torchvision.transforms as T

from PIL import Image
import numpy as np
import re
import pandas as pd
import nibabel as nib
import glob
import sys
import skimage.transform as skTrans
import torchio as tio

join = os.path.join

class Dataset_3D(Dataset):
    def __init__(self,transforms="",transforms_xr="",root="",type=1):
        labels_file_ct=""
        if type==0:
            labels_file_ct = join( r"./", r"Prova_All_Train_Sub.csv")
        elif type==1:
            labels_file_ct = join( r"./", r"Prova_All_Val_Sub.csv")
        elif type==2:
            labels_file_ct = join( r"./", r"Prova_All_Test_Sub.csv")

        self.type = type
        self.transforms_nii= eval(transforms or Compose([]))
        self.transforms_xr = eval(transforms_xr or Compose([]))
        self.paths=[]
        self.names=[]
        self.age=[]
        self.gender=[]
        self.extra=[]
        self.y=[]
        if self.type>=0:
            labels_file = pd.read_csv(labels_file_ct,sep=r',', engine='python')
        self.n_patients = len(labels_file)
        print("##N_Patiens = " + str(self.n_patients))
        self.patient_idx = np.arange(self.n_patients)
        labels={}
        for _, row in labels_file.iterrows():
            labels[row["id"]]=row["result"]
        path= root+"/"
        for _,x in labels_file.iterrows():
            if(self.type>=0):
                self.paths.append([path + str(x["path"]), path + str(x["X_1"]), path + str(x["X_2"])])
            self.session= x["id"]
            funct = lambda num: 1 if num=="POSITIVO" else 0
            self.y.append(funct(x["result"]))

    def normalize(self,item):
        for i in range(item.size()[0]):
            minval = item[i,...].min()
            maxval = item[i,...].max()
            if minval != maxval:
                item[i,...] -= minval
                item[i,...] *= (2/ (maxval - minval))
                item[i,...] -= 1
        return item

    def __getitem__(self, item):
        exists =np.array([0,0,0])
        path=self.paths[item]
        if os.path.exists(path[0]+".npz"):
            ct=np.load(path[0]+".npz")
            ct = ct['arr_0']
            exists[0]=1
        elif os.path.exists(path[0]) and re.search(r".*npz", path[0]):
            ct = np.load(path[0])
            ct = ct['arr_0']
            exists[0] = 1
        else:
            ct=np.zeros([250, 250, 200], dtype=float)
            
        x,y,z=ct.shape
        if not(x==250) or not(y==250) or not(z==200):
            print(path[0])
            print(ct.shape)
            time.sleep(10)

        ct = ct[None, ...]
        ct=torch.from_numpy(ct).type(torch.FloatTensor)
        X_1 = torch.unsqueeze(torch.from_numpy(np.zeros([2048, 2048], dtype=float)),0).type(torch.FloatTensor)
        X_2 = torch.unsqueeze(torch.from_numpy(np.zeros([2048, 2048], dtype=float)),0).type(torch.FloatTensor)
        if (self.type >= 0):
            if os.path.exists(path[1]):
                X_1= Image.open(path[1])
                X_1 = self.normalize(self.transforms_xr(X_1))
            if os.path.exists(path[2]):
                X_2 = Image.open(path[2])
                X_2 = self.normalize(self.transforms_xr(X_2))

        y=torch.as_tensor(self.y[item], dtype=torch.uint8)
        session=self.session
        return ct,y,session,X_1,X_2 #float(self.age[item]),float(self.gender[item]=='M'),exists

    def __len__(self):
        return self.n_patients







