import os
import torch
import time
from torch.utils.data import Dataset, DataLoader

import data.transforms as dt
import torch.nn.functional as f
from torchvision.transforms import Resize,RandomAffine
from torchvision.transforms import Compose,PILToTensor
import torchvision.transforms as T

from PIL import Image
import numpy as np
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
            labels_file_ct = join(root,"BIMCV-COVID19-cIter_1_2\images\Labels", "Try_Train.csv")
        elif type==1:
            labels_file_ct = join(root,"BIMCV-COVID19-cIter_1_2\images\Labels", "ct_sessions.csv")
        elif type==2:
            labels_file_ct = join(root, "BIMCV-COVID19-cIter_1_2\images\Labels", "Try_Test.csv")

        #labels_file_xr = join(root,"BIMCV-COVID19-cIter_1_2\images\Labels","rx_sessions.csv")
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
        elif self.type<0:
            labels_file = pd.read_csv(labels_file_xr,sep=r',', engine='python')
        self.n_patients = len(labels_file)
        print("##N_Patiens = " + str(self.n_patients))
        self.patient_idx = np.arange(self.n_patients)
        #self.labels_file_xr=pd.read_csv(labels_file_xr,sep=r',', engine='python')
        labels={}
        for _, row in labels_file.iterrows():
            labels[row["session_id"]]=row["result"]
        path= root+"/"
        for _,x in labels_file.iterrows():
            if(self.type==0 or type==2):
                '''if not os.path.exists(path+x["path"]):
                    x["path"] = os.path.join("E:/",x["path"])
                    self.paths.append([x["path"],path + str(x["X_1"]),path + str(x["X_2"])])

                else:
                    self.paths.append([path + x["path"],path + str(x["X_1"]),path + str(x["X_2"])])'''
                self.paths.append([path + str(x["path"]), path + str(x["X_1"]), path + str(x["X_2"])])

            else:
                self.paths.append(x["path"])#Add double x-ray images
            self.names.append(x["sub"])
            self.age.append(x["age"])
            self.gender.append(x["sex"])
            self.session= x["session_id"]
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
        name=self.names[item]
        if os.path.exists(path[0]+".npz"):
            ct=np.load(path[0]+".npz")
            ct = ct['arr_0']
            exists[0]=1
        else:
            ct=np.zeros([250, 250, 200], dtype=float)

        ct = ct[None, ...]
        ct=torch.from_numpy(ct).type(torch.FloatTensor)
        #transforms = RandomAffine(degrees=10)
        #ct = transforms(ct)
        #ct = ct_trans(ct)
        #ct = torch.unsqueeze(torch.from_numpy(np.zeros([120, 120, 120], dtype=float)), 0).type(torch.FloatTensor)
        X_1 = torch.unsqueeze(torch.from_numpy(np.zeros([2048, 2048], dtype=float)),0).type(torch.FloatTensor)
        X_2 = torch.unsqueeze(torch.from_numpy(np.zeros([2048, 2048], dtype=float)),0).type(torch.FloatTensor)
        if (self.type >= 0):
            if os.path.exists(path[1]):
                exists[1]=1
                X_1= Image.open(path[1])
                X_1 = self.normalize(self.transforms_xr(X_1))
            if os.path.exists(path[2]):
                exists[2]=1
                X_2 = Image.open(path[2])
                X_2 = self.normalize(self.transforms_xr(X_2))

        y=torch.as_tensor(self.y[item], dtype=torch.uint8)
        session=self.session
        return ct,y,name,session,X_1,X_2,float(self.age[item]),float(self.gender[item]=='M'),exists

    def __len__(self):
        return self.n_patients







