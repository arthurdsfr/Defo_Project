import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, num_classes, BatchNorm):
        super(Decoder, self).__init__()
        self.num_classes = num_classes
        self.norm = BatchNorm
        self.dropout = 0.1

        # Define all layers here (with guessed input channels for skip connections)
        self.dec1 = self.decoder_conf(512, 512, 3, 1, self.norm, self.dropout)  # from middle
        self.dec2 = self.decoder_conf(512 + 256, 256, 2, 2, self.norm, self.dropout)
        self.dec3 = self.decoder_conf(256, 256, 3, 1, self.norm, self.dropout)

        self.dec4 = self.decoder_conf(256 + 128, 128, 2, 2, self.norm, self.dropout)
        self.dec5 = self.decoder_conf(128, 128, 3, 1, self.norm, self.dropout)

        self.dec6 = self.decoder_conf(128 + 96, 64, 2, 2, self.norm, self.dropout)
        self.dec7 = self.decoder_conf(64, 64, 5, 1, self.norm, self.dropout)

        self.final = self.decoder_conf(64, num_classes, 1, 1, '', self.dropout, 1.0, 0.02)

    def decoder_conf(self, in_channels, out_channels, f_size, scale, norm, dropout=0.0, stddev=-1.0, slope=0.00):
        layers = []
        if scale > 1:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=f_size, stride=scale, padding=0))
        else:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=f_size, stride=1, padding=0))

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

    def forward(self, features):
        # Unpack encoder features
        x_last = features[-1]  # 512
        x1 = features[-2]  # 256
        x0 = features[-3]  # 128
        x_early = features[0]  # 96

        x = self.dec1(x_last)
        x = torch.cat((x, x1), dim=1)
        x = self.dec2(x)
        x = self.dec3(x)
        x = torch.cat((x, x0), dim=1)
        x = self.dec4(x)
        x = self.dec5(x)
        x = torch.cat((x, x_early), dim=1)
        x = self.dec6(x)
        x = self.dec7(x)
        x = self.final(x)
        return x