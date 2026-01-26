import argparse
from options.baseoptions import BaseOptions

class DeepLabOptions():
    def __init__(self, parser):
        super(DeepLabOptions, self).__init__()
        self.parser = parser
    def initialize(self):
        #self.parser = BaseOptions(self.parser).initialize()
        self.parser.add_argument('--deeplaboutput_stride', default=16, type=int, help='Patch resolution of the model.')
        self.parser.add_argument("--deeplabbackbone", default = 'resnet', type = str, help = 'Dino feature extractor architecture')
        self.parser.add_argument("--deeplab_imagenet_pretrain",type = eval, choices = [True, False], default = False, help = "Defines if the dino feature extractor will be taken from imagenet or not")
        return self.parser