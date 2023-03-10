import torch
import torch.nn as nn




class conv3d_norm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in'):
        super(conv3d_norm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = nn.BatchNorm3d(in_ch)
        self.activation = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x

class conv2d_norm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros'):
        super(conv2d_norm, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)
        self.norm = nn.BatchNorm2d(in_ch)
        self.activation = nn.ReLU(inplace=True)



    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.activation(x)
        return x







