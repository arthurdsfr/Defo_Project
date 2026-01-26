import os
import argparse

class BaseOptions():
    def __init__(self, parser):
        self.parser = parser
    def initialize(self):
        self.parser.add_argument('--featureextractor_arch', default='deeplab', type=str, help='Feature extractor Architecture')
        self.parser.add_argument('--segmentationhead_arch', default = '', type = str, help = 'Segmentation Head Architecture')
        self.parser.add_argument('--input_patch_size', default = 64, type = int, help = 'Size of extracted patches from input images')
        self.parser.add_argument('--dataset_name', default = "deforestation", type = str, help = 'Dataset name for supervised training')
        self.parser.add_argument('--background', type = eval, choices = [True, False], default = True, help = 'Decides if the model will be trained with background or not')
        self.parser.add_argument('--num_classes', default=2, type=int, help='Number of labels for linear classifier')
        self.parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
        self.parser.add_argument('--multiple_gpus', type = eval, choices = [True, False], default = False, help = 'Decides if the model will be trained with multi-gpu or not')
        self.parser.add_argument('--data_path', default='/media/psoto/Dados/Pedro/WORK/PEDRO/DDETECTION/DATA/', type=str)
        self.parser.add_argument('--csv_path', default='')
        self.parser.add_argument('--fe_pretrained_weights', default="None", type=str, help="Path to pretrained weights to train.")
        self.parser.add_argument('--sh_pretrained_weights', default="None", type=str, help="Path to pretrained weights to train")
        self.parser.add_argument("--freeze_featureextractor",type = eval, choices = [True, False], default = False, help = "Defines if the feature extractor will be taken freeze or not")
        self.parser.add_argument('--experiment_mainpath', type = str, default = '/media/psoto/LV-02/Pedro/EXPERIMENTS/')
        self.parser.add_argument("--overall_projectname", type = str, default = "/DDETECTION", help = "Set the Global project name")
        self.parser.add_argument("--experiment_name", type = str, default = "/AMAZON_PA_DEEPLAB_CENTERCROP_ADAM_COSINE", help = "Set the name of the expriment")

        return self.parser