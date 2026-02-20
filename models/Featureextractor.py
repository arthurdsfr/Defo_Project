import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import dino.vision_transformer as vits
from collections import OrderedDict
from utils import *
from deeplab.deeplab import DeepLab
from vnet.vnet import VNet


def _safe_torch_load(path, map_location="cpu"):
    # PyTorch >=2.6 defaults to weights_only=True, which can fail for full checkpoints.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Backward compatibility for older PyTorch versions.
        return torch.load(path, map_location=map_location)

class DinoFeaturizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dim = 70
        patch_size = self.args.dino_patch_size
        self.patch_size = patch_size
        #self.feat_type = self.cfg.dino_feat_type
        arch = self.args.dino_arch

        self.model = vits.__dict__[arch](
            patch_size=patch_size,
            in_chans = self.args.input_nc,
            num_classes=0)
        
        if not self.args.freeze_featureextractor:
            for p in self.model.parameters():
                p.requires_grad = False
        
        self.dropout = torch.nn.Dropout2d(p=.1)
        if self.args.dino_imagenet_pretrain:
            if arch == "vit_small" and patch_size == 16:
                url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif arch == "vit_small" and patch_size == 8:
                url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
            elif arch == "vit_base" and patch_size == 16:
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif arch == "vit_base" and patch_size == 8:
                url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            else:
                raise ValueError("Unknown arch and patch size")
        print(args.fe_pretrained_weights)
        if args.fe_pretrained_weights != "None":
            state_dict = _safe_torch_load(self.args.fe_pretrained_weights, map_location="cpu")
            if self.args.phase == 'test' and self.args.freeze_featureextractor:
                state_dict = state_dict["teacher"]
            elif self.args.phase == 'train':
                state_dict = state_dict["teacher"]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            #state_dict = {k.replace("prototypes", "last_layer"): v for k, v in state_dict.items()}
            new_state_dict = OrderedDict()
            for k,v in state_dict.items():
                if 'head.' not in k:
                    new_state_dict[k] = v
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(self.args.fe_pretrained_weights, msg))
        elif self.args.dino_imagenet_pretrain:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)
        else:
            print("Since no pretrained weights have been provided and pretrain option is false, we will train DINO from scratch.")

        
        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768

    def forward(self, img ,n=1):
        
        assert (img.shape[2] % self.patch_size == 0)
        assert (img.shape[3] % self.patch_size == 0)
       
        # get selected layer activations
        if len(n) > 1:
            image_feat = []
            if self.args.segmentationhead_arch == 'unetr':
                image_feat.append(img)
            feat_, attn_, qkv_ = self.model.get_specific_intermediate_feat(img, n=n)

            for i in range(len(n)):
                feat, attn, qkv = feat_[i], attn_[i], qkv_[i]
    
                feat_h = img.shape[2] // self.patch_size
                feat_w = img.shape[3] // self.patch_size

        
                image_feat.append(feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2))
        else:
            if n == 1:
                feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
                feat, attn, qkv = feat[0], attn[0], qkv[0]
        
                feat_h = img.shape[2] // self.patch_size
                feat_w = img.shape[3] // self.patch_size

        
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif n > 1:
                image_feat = []
                feat_, attn_, qkv_ = self.model.get_intermediate_feat(img, n=n)
                
                for i in range(n):
                    feat, attn, qkv = feat_[i], attn_[i], qkv_[i]
                    feat_h = img.shape[2] // self.patch_size
                    feat_w = img.shape[3] // self.patch_size

            
                    image_feat.append(feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2))

        return image_feat

class DeepLabFeaturizer(nn.Module):
    def __init__(self, args):
        super(DeepLabFeaturizer, self).__init__()
        self.args = args
        self.output_stride = self.args.deeplaboutput_stride
        self.backbone = self.args.deeplabbackbone
        self.imagenet_pretrain = self.args.deeplab_imagenet_pretrain
        self.model = DeepLab(backbone=self.backbone, output_stride=self.output_stride, input_channles = self.args.input_nc, pretrained = self.imagenet_pretrain)
        if args.fe_pretrained_weights != "None":
            state_dict = _safe_torch_load(self.args.fe_pretrained_weights, map_location="cpu")

            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            # remove `model.` prefix
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

            msg = self.model.load_state_dict(state_dict, strict=True)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(self.args.fe_pretrained_weights, msg))
    
    def forward(self, img):
        features = []
        image_feat, low_level_features = self.model(img)
        features.append(low_level_features)
        features.append(image_feat)
        return features
    
class VnetFeaturizer(nn.Module):
    def __init__(self, args):
        super(VnetFeaturizer, self).__init__()
        self.args = args
        self.model = VNet(self.args)

        if args.fe_pretrained_weights != "None":
            state_dict = _safe_torch_load(self.args.fe_pretrained_weights, map_location="cpu")

            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            # remove `model.` prefix
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

            msg = self.model.load_state_dict(state_dict, strict=True)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(self.args.fe_pretrained_weights, msg))
    
    def forward(self, img):
        features = []
        x_last, x0, x1, x2 = self.model(img)
        features.append(x0) 
        features.append(x1)
        features.append(x2)
        features.append(x_last)
        return features

