import argparse
from options.baseoptions import BaseOptions

class DinoOptions():
    def __init__(self, parser):
        super(DinoOptions, self).__init__()
        self.parser = parser
    def initialize(self):
        #self.parser = BaseOptions(self.parser).initialize()
        self.parser.add_argument('--dino_intermediate_blocks', default=[2, 5, 8, 11], type=int, help="""Concatenate [CLS] tokens for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
        self.parser.add_argument('--dino_patch_size', default=16, type=int, help='Patch resolution of the model.')
        self.parser.add_argument("--dino_checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
        self.parser.add_argument("--dino_arch", default = 'vit_small', type = str, help = 'Dino feature extractor architecture')
        self.parser.add_argument("--dino_imagenet_pretrain",type = eval, choices = [True, False], default = True, help = "Defines if the dino feature extractor will be taken from imagenet or not")
        
        return self.parser