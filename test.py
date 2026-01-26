import os
import json
import glob
import argparse
from datetime import datetime
from argparse import Namespace

import utils

from data.DeforestationDataset import *
from models.Models import *
from options.testoptions import TestOptions
from options.deforestationoptions import *

def main():
    args = TestOptions().initialize()
    args.device_number = torch.cuda.device_count()
    #Saving path for the segmenation checkpoint
    args.checkpoint_savepath = args.experiment_mainpath + args.overall_projectname + args.experiment_name + "/checkpoints"
    if not os.path.exists(args.checkpoint_savepath):
        print("The current folder: " + args.checkpoint_savepath + "doesn't exists")
        print("Please, make sure you are addressing the right checkpoint folders")
        sys.exit()
    
    training_folders = os.listdir(args.checkpoint_savepath)
    if len(training_folders) == 0:
        print("The current folder: " + args.checkpoint_savepath + "doesn't contains trained models")
        print("Please, make sure you are addressing the right checkpoint folders")
        sys.exit()
    else:
        trainedweights_files = glob.glob(args.checkpoint_savepath + '/**/*.pth', recursive = True)
        print(f"{len(trainedweights_files)} .pth files found in " + args.checkpoint_savepath + " directory.")
        if len(trainedweights_files) == 0:
            print("No trained weight were stored in this address")
            sys.exit()
        else:
            args.results_savepath = args.experiment_mainpath + args.overall_projectname + args.experiment_name + "/results/"
            if not os.path.exists(args.results_savepath):
                os.makedirs(args.results_savepath)
    
    args.numberpth_rest = len(trainedweights_files)%2
    print("Creating the custom dataset")
    
    if args.dataset_name == 'deforestation':
        dataset = DeforestationDataset(args)
    
    if args.numberpth_rest != 0:
        for trainedweight_file in trainedweights_files:
            args.results_savepath = args.experiment_mainpath + args.overall_projectname + args.experiment_name + "/results/"
            training_folder = trainedweight_file.split("/")[-2]
            args.sh_pretrained_weights = trainedweight_file
            args.results_savepath = args.results_savepath + training_folder + "/"
            if not os.path.exists(args.results_savepath):
                os.makedirs(args.results_savepath)
            model = Models(args, dataset)
            model.evaluate()
    else:
        for training_folder in training_folders:
            # includes a print for signalizes which folder we are working with
            print(training_folder)
            args.results_savepath = args.experiment_mainpath + args.overall_projectname + args.experiment_name + "/results/"
            args.results_savepath = args.results_savepath + training_folder + "/"
            args.sh_pretrained_weights = args.checkpoint_savepath + '/' + training_folder + '/segmentation_head.pth'
            args.fe_pretrained_weights = args.checkpoint_savepath + '/' + training_folder + '/feature_extractor.pth'
            if not os.path.exists(args.results_savepath):
                os.makedirs(args.results_savepath)
            model = Models(args, dataset)
            model.evaluate()

if __name__ == "__main__":
    main()