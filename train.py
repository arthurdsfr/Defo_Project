import os
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from argparse import Namespace


from data.DeforestationDataset import *
from models.Models import *
from options.trainoptions import TrainOptions
from options.deforestationoptions import *
from utils.tools import *

def main():
    args = TrainOptions().initialize()
    args.device_number = torch.cuda.device_count()

    #Saving path for the segmenation checkpoint
    args.checkpoint_savepath_ = args.experiment_mainpath + args.overall_projectname + args.experiment_name + "/checkpoints"
    os.makedirs(args.checkpoint_savepath_, exist_ok=True)

    print("Creating the custom dataset")
    
    if args.dataset_name == 'deforestation':
        dataset = DeforestationDataset(args)
    
    for r in range(args.runs):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        args.checkpoints_savepath = args.checkpoint_savepath_ + "/model_" + dt_string + "/"
        os.makedirs(args.checkpoints_savepath, exist_ok=True)
        with open(args.checkpoints_savepath + 'commandline_args.txt', 'w') as f:
              for i in args.__dict__:
                print(str(i) + ": " ,getattr(args, i))
                f.write(str(i) + ": " + str(getattr(args, i)) + "\n")
        
        model = Models(args, dataset)
        model.train()

if __name__ == "__main__":
    main()