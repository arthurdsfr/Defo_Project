from deeplab.backbones import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm, input_channels, pretrained):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm, input_channels=input_channels, pretrained=pretrained)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, input_channels = input_channels, pretrained=False)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError