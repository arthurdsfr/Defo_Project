import torch
import torch.nn as nn
import torch.nn.functional as F
from deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class VNet(nn.Module):
    def __init__(self, args, input_channels = 14):
        super(VNet, self).__init__()
        self.args = args
        self.args.norm = 'B'
        self.args.dropout = 0.1

        # Define all your encoder layers here
        self.enc0 = self.encoder_conf(input_channels, 96, 5, 1, self.args.norm, self.args.dropout)
        self.enc1 = self.encoder_conf(96, 96, 2, 2, self.args.norm, self.args.dropout)
        self.enc2 = self.encoder_conf(96, 128, 3, 1, self.args.norm, self.args.dropout)
        self.enc3 = self.encoder_conf(128, 128, 2, 2, self.args.norm, self.args.dropout)
        self.enc4 = self.encoder_conf(128, 256, 3, 1, self.args.norm, self.args.dropout)
        self.enc5 = self.encoder_conf(256, 256, 2, 2, self.args.norm, self.args.dropout)
        self.enc6 = self.encoder_conf(256, 512, 3, 1, self.args.norm, self.args.dropout)

    def encoder_conf(self, in_channels, out_channels, f_size, scale, norm, dropout=0.0, stddev=-1.0, slope=0.00):
        layers = []
        if scale > 1:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=f_size, stride=scale, padding=0))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=f_size, stride=1, padding=0))

        if norm == 'I':
            layers.append(nn.InstanceNorm2d(out_channels))
        elif norm == 'B':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == 'G':
            layers.append(nn.GroupNorm(16, out_channels))

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        if slope < 1.0:
            layers.append(nn.LeakyReLU(negative_slope=slope) if slope > 0.0 else nn.ReLU())
        else:
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, input):
        X = self.enc0(input)
        X0 = self.enc1(X)
        X = self.enc2(X0)
        X_EARLY = X
        X1 = self.enc3(X)
        X = self.enc4(X1)
        X2 = self.enc5(X)
        X = self.enc6(X2)
        X_MIDDLE = X
        return X, X0, X1, X2