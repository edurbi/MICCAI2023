import time

import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import conv3d_norm, conv2d_norm
import numpy as np
import monai as moi
from resnet_3D import generate_model
from monai.networks.nets.resnet import ResNetBlock
import torchvision.transforms as T


basic_dims = 2
mlp_dim = 4098
num_heads = 8
patch_size = 8
class Encoder(nn.Module):
    def __init__(self, in_dim):
        super(Encoder, self).__init__()

        self.begining = nn.Conv3d(in_channels=1, out_channels=in_dim, kernel_size=3, stride=1, padding=1,
                               padding_mode='reflect', bias=True)
        self.begining1 = nn.MaxPool3d(2)
        self.e1_c1 = nn.Conv3d(in_channels=in_dim, out_channels=in_dim*2, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=True)
        self.m1 = nn.MaxPool3d(2)
        self.e1_c2 = conv3d_norm(in_dim*2, in_dim*2, pad_type='reflect')
        self.e2_c1 = conv3d_norm(in_dim*2, in_dim*4, stride=2, pad_type='reflect')
        self.e2_c2 = conv3d_norm(in_dim*4, in_dim*4, pad_type='reflect')
        self.e3_c1 = conv3d_norm(in_dim*4, in_dim*8, stride=1, pad_type='reflect')
        self.e3_c2 = conv3d_norm(in_dim*8, in_dim*8, pad_type='reflect')
        self.e4_c1 = conv3d_norm(in_dim*8, in_dim*16, stride=2, pad_type='reflect')
        self.e4_c2 = conv3d_norm(in_dim*16, in_dim*16, pad_type='reflect')
        self.e5_c1 = conv3d_norm(in_dim*16, in_dim*32, stride=1, pad_type='reflect')
        self.e5_c2 = conv3d_norm(in_dim*32, in_dim*32, pad_type='reflect')

    def forward(self, x):
        x= self.begining(x)
        x = self.begining1(x)
        x1 = self.e1_c1(x)
        #x1= self.m1(x1)
        x1 = x1 + self.e1_c2(x1)
        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c2(x2)
        x3 = self.e3_c1(x2)
        x3 = x3 + (self.e3_c2(x3))
        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c2(x4)
        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c2(x5)

        return x5

class Encoder2d(nn.Module):
    def __init__(self):
        super(Encoder2d, self).__init__()

        self.e1_c1 = nn.Conv2d(in_channels=1, out_channels=basic_dims, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=True)
        self.e1_c2 = conv2d_norm(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = conv2d_norm(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = conv2d_norm(basic_dims, basic_dims*2, stride=2, pad_type='reflect')
        self.e2_c2 = conv2d_norm(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e2_c3 = conv2d_norm(basic_dims*2, basic_dims*2, pad_type='reflect')

        self.m1 = nn.MaxPool2d(2, stride=2)

        self.e3_c1 = conv2d_norm(basic_dims*2, basic_dims*4, stride=2, pad_type='reflect')
        self.e3_c2 = conv2d_norm(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e3_c3 = conv2d_norm(basic_dims*4, basic_dims*4, pad_type='reflect')

        self.e4_c1 = conv2d_norm(basic_dims*4, basic_dims*8, stride=2, pad_type='reflect')
        self.e4_c2 = conv2d_norm(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e4_c3 = conv2d_norm(basic_dims*8, basic_dims*8, pad_type='reflect')

        self.m2 = nn.MaxPool2d(2, stride=2)

        self.e5_c1 = conv2d_norm(basic_dims*8, basic_dims*16, stride=2, pad_type='reflect')
        self.e5_c2 = conv2d_norm(basic_dims*16, basic_dims*16, pad_type='reflect')
        self.e5_c3 = conv2d_norm(basic_dims*16, basic_dims*16, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))
        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))
        x2=self.m1(x2)
        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))
        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))
        x4= self.m2(x4)
        '''x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))'''
        x4 = x4.view(-1,1,256,256)
        return x4

class Decoder_fuse(nn.Module):
    def __init__(self):
        super(Decoder_fuse, self).__init__()
        self.fc1 = nn.Linear(256, 64,dtype=torch.float32)
        self.fc3 = nn.Linear(64, 2,dtype=torch.float32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x




class Model(nn.Module):
    def __init__(self, num_cls=1):
        super(Model, self).__init__()
        self.in_mod=1
        self.flair_encoder = Encoder(self.in_mod)
        self.flair_transform = moi.networks.nets.vit.ViT(in_channels=self.in_mod*32,img_size=(32,32,25), pos_embed='conv',patch_size=(8,8,6),hidden_size=1024,num_heads=8,num_layers = 8)
        self.X_1 = moi.networks.nets.vit.ViT(in_channels=2,img_size=(256,256), pos_embed='conv',patch_size=(16,16),spatial_dims=2,hidden_size=256,num_heads=16,num_layers = 12)
        self.X_2 = moi.networks.nets.vit.ViT(in_channels=2,img_size=(256,256), pos_embed='conv',patch_size=(16,16),spatial_dims=2,hidden_size=256,num_heads=16,num_layers = 12)
        self.X_3 = moi.networks.nets.vit.ViT(in_channels=2, img_size=(256, 256), pos_embed='conv',
                                             patch_size=(16, 16),
                                             spatial_dims=2, hidden_size=256, num_heads=16, num_layers=12)
        self.final1 = moi.networks.nets.resnet.resnet18(spatial_dims=2, num_classes=2,n_input_channels=6)
        '''self.final1 = moi.networks.nets.vit.ViT(in_channels=6, img_size=(256, 256), pos_embed='conv',
                                             patch_size=(16, 16),
                                             spatial_dims=2, hidden_size=256, num_heads=16, num_layers=12, classification=True
                                            ,num_classes=2)'''

        self.reducer = Encoder2d()
        self.reducer2 = Encoder2d()
        self.is_training = False


    def random_elimination(self,matrix,prob):
        batch,_,x,y=matrix.size()
        if torch.rand(1)>0.6:
            prob=0
        prob = torch.rand(1)* prob
        mask=(torch.rand(x//16, y//16,dtype=torch.float32)>=prob).cuda()
        mask = mask.expand(1, 1, x//16, y//16).to(torch.float32)
        mask= F.interpolate(mask,scale_factor=16,mode='nearest-exact')
        mask=mask.repeat(batch,1, 1, 1)
        matrix = torch.mul(matrix,mask)
        return matrix

    def forward(self, x, batch,rad_1,rad_2,age,gender,training=True,exists=[1,1,1]):

        featuresx = self.flair_encoder(x)
        featurest,_=self.flair_transform(featuresx)
        featurest=featurest.view(-1,256,256)
        flair_x5 = featurest[None, ...].permute(1, 0, 2, 3)
        if training:
            flair_x5=self.random_elimination(flair_x5,1)

        rad_1 = self.reducer(rad_1)
        rad_2 = self.reducer2(rad_2)
        rad_1f = torch.cat((rad_1,flair_x5),1)
        rad_2f = torch.cat((rad_2, flair_x5), 1)
        rad_3f = torch.cat((rad_1, rad_2), 1)
        conv_X1,_= self.X_1(rad_1f)
        conv_X2,_ = self.X_2(rad_2f)
        conv_X3,_ = self.X_3(rad_3f)
        conv_X3 = conv_X3[None, ...].permute(1, 0, 2, 3)
        conv_X2 = conv_X2[None,...].permute(1, 0, 2, 3)
        conv_X1 = conv_X1[None, ...].permute(1, 0, 2, 3)
        if training :
            conv_X1 = self.random_elimination(conv_X1,1)
        if training :
            conv_X2 = self.random_elimination(conv_X2, 1)
        if training:
            conv_X3 = self.random_elimination(conv_X3,1)


        x_final = torch.cat((conv_X2, conv_X1, flair_x5, rad_1, rad_2,conv_X3), dim=1)
        preds,_ = self.final1(x_final)
        return preds
