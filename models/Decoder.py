import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from deeplab.decoder import build_decoder
from deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from vnet.decoder import Decoder

class BasicDecoder(nn.Module):
    def __init__(self, args):
        super(BasicDecoder, self).__init__()
        self.args = args
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.args.input_nc, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, self.args.output_nc, 3, stride=2, padding=1, output_padding=1),
        )
        self.softmax = nn.Softmax(dim = 1)
        
        if self.args.sh_pretrained_weights is not None:
            new_state_dict = OrderedDict()
            state_dict = torch.load(self.args.sh_pretrained_weights, map_location="cpu")
            state_dict = {k.replace("decoder.", ""): v for k, v in state_dict.items()}
            msg = self.decoder.load_state_dict(new_state_dict, strict=False)
            print('Segmentation Head weights found at {} and loaded with msg: {}'.format(self.args.sh_pretrained_weights, msg))
        
    def forward(self, input):
        logits = self.decoder(input)
        predit = self.softmax(logits)
        return logits, predit
    
class DeepLabDecoder(nn.Module):
    def __init__(self, args, sync_bn=False, freeze_bn=False):
        super(DeepLabDecoder, self).__init__()
        self.args = args
        num_classes = self.args.output_nc 
        backbone = self.args.deeplabbackbone
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.softmax = nn.Softmax(dim = 1)

        if self.args.sh_pretrained_weights != "None":
            state_dict = torch.load(self.args.sh_pretrained_weights, map_location="cpu")
            # remove `decoder.` prefix
            state_dict = {k.replace("decoder.", ""): v for k, v in state_dict.items()}
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            msg = self.decoder.load_state_dict(state_dict, strict=True)
            print('Segmentation Head weights found at {} and loaded with msg: {}'.format(self.args.sh_pretrained_weights, msg))


    def forward(self, img):
        low_level_feat = img[0]
        img = img[-1]
        logits = self.decoder(img, low_level_feat)
        logits = F.interpolate(logits, size=[self.args.input_patch_size, self.args.input_patch_size], mode='bilinear', align_corners=True)
        predit = self.softmax(logits)
        return logits, predit

class UneTr(nn.Module):
    def __init__(self, args):
        super(UneTr, self).__init__() 
        self.args = args
        num_classes = self.args.output_nc
        input_nc = self.args.input_nc
        image_nc = self.args.image_nc

        self.green_conv_b0 = self.green_block(input_nc, 512)
        self.green_conv_b1 = self.green_block(512, 256)
        self.green_conv_b2 = self.green_block(256, 128)
        self.green_conv_b3 = self.green_block(128, 64)
        
        self.blue_conv_b0_0 = self.blue_block(input_nc, 512)
        self.blue_conv_b1_0 = self.blue_block(input_nc, 256)
        self.blue_conv_b1_1 = self.blue_block(256, 256)
        self.blue_conv_b2_0 = self.blue_block(input_nc, 128)
        self.blue_conv_b2_1= self.blue_block(128, 128)
        self.blue_conv_b2_2= self.blue_block(128, 128)

        self.yellow_conv_b0 = self.yellow_block(1024, 512)
        self.yellow_conv_b1 = self.yellow_block(512, 256)
        self.yellow_conv_b2 = self.yellow_block(256, 128)
        self.yellow_conv_b3_1 = self.yellow_block(image_nc, 64)
        self.yellow_conv_b3_2 = self.yellow_block(128, 64)

        self.output = nn.Conv2d(64, num_classes, 1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim = 1)
        
    def green_block(self, input_nc, output_nc):
        return nn.ConvTranspose2d(input_nc, output_nc, 2, stride=2, padding=0, output_padding=0)
    def blue_block(self, input_nc, output_nc):
        return nn.Sequential(nn.ConvTranspose2d(input_nc, output_nc, 2, stride=2, padding=0, output_padding=0),
                             nn.Conv2d(output_nc, output_nc, 3, stride=1, padding=1),
                             nn.BatchNorm2d(num_features=output_nc),
                             nn.ReLU())
    def yellow_block(self, input_nc, output_nc):
        return nn.Sequential(nn.Conv2d(input_nc, output_nc, 3, stride=1, padding=1),
                             nn.BatchNorm2d(num_features=output_nc),
                             nn.ReLU(),
                             nn.Conv2d(output_nc, output_nc, 3, stride=1, padding=1),
                             nn.BatchNorm2d(num_features=output_nc),
                             nn.ReLU()
                             )

    def forward(self, x):
        z_l = x[-1]
        z_9 = x[-2]
        z_6 = x[-3]
        z_3 = x[-4]
        input = x[-5]

        z_9 = self.blue_conv_b0_0(z_9)

        z_6 = self.blue_conv_b1_0(z_6)
        z_6 = self.blue_conv_b1_1(z_6)
        
        z_3 = self.blue_conv_b2_0(z_3)
        z_3 = self.blue_conv_b2_1(z_3)
        z_3 = self.blue_conv_b2_2(z_3)

        z_0 = self.yellow_conv_b3_1(input)
        # Upsample 0
        g_l = self.green_conv_b0(z_l)
        concat_1 = torch.cat((g_l, z_9), dim=1)
        deconv_1 = self.yellow_conv_b0(concat_1)
        # Upsample 1
        g_9 = self.green_conv_b1(deconv_1)
        concat_2 = torch.cat((g_9, z_6), dim=1)
        deconv_2 = self.yellow_conv_b1(concat_2)
        # Upsample 2
        g_6 = self.green_conv_b2(deconv_2)
        concat_3 = torch.cat((g_6, z_3), dim=1)
        deconv_3 = self.yellow_conv_b2(concat_3)
        # Upsample 3
        g_3 = self.green_conv_b3(deconv_3)
        concat_4 = torch.concat((g_3, z_0), dim = 1)
        deconv_4 = self.yellow_conv_b3_2(concat_4)
        
        logits = self.output(deconv_4)
        predit = self.softmax(logits)
        return logits, predit   

class UneTrDecoder(nn.Module):
    def __init__(self, args):
        super(UneTrDecoder, self).__init__() 
        self.args = args
        self.decoder = UneTr(self.args)
        if self.args.sh_pretrained_weights != "None":
            state_dict = torch.load(self.args.sh_pretrained_weights, map_location="cpu")
            state_dict = {k.replace("decoder.", ""): v for k, v in state_dict.items()}
            msg = self.decoder.load_state_dict(state_dict, strict=True)
            print('Segmentation Head weights found at {} and loaded with msg: {}'.format(self.args.sh_pretrained_weights, msg))
    def forward(self, x):
        logits, predit = self.decoder(x)
        return logits, predit
    
class VnetDecoder(nn.Module):
    def __init__(self, args):
        super(VnetDecoder, self).__init__()
        self.args = args
        num_classes = self.args.output_nc
        self.decoder = Decoder(num_classes, 'B')
        self.softmax = nn.Softmax(dim = 1)
        if self.args.sh_pretrained_weights != "None":
            state_dict = torch.load(self.args.sh_pretrained_weights, map_location="cpu")
            # remove `decoder.` prefix
            state_dict = {k.replace("decoder.", ""): v for k, v in state_dict.items()}
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            msg = self.decoder.load_state_dict(state_dict, strict=True)
            print('Segmentation Head weights found at {} and loaded with msg: {}'.format(self.args.sh_pretrained_weights, msg))
    
    def forward(self, img):
        logits = self.decoder(img)
        predit = self.softmax(logits)
        return logits, predit

        