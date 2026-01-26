import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from progress.bar import Bar

from utils.tools import *
from options.testoptions import TestOptions
def main():
    args = TestOptions().initialize()
    if 'ifremer' in args.dataset_name:
        args.domain = args.domain_ifremer
    
    args.results_savepath = args.experiment_mainpath + args.overall_projectname + args.experiment_name + "/results/"
    if not os.path.exists(args.results_savepath):
        print("The current folder: " + args.results_savepath + "doesn't exists")
        print("Please, make sure you are addressing the right checkpoint folders")
        sys.exit()
    
    results_folders = os.listdir(args.results_savepath)
    if len(results_folders) == 0:
        print("The current folder: " + args.results_folders + "doesn't contains evaluated moodels")
        print("Please, make sure you are addressing the right results folders")
        sys.exit()

    for result_folder in results_folders:
        if "domain" in args:
            resultfolder_path = args.results_savepath + result_folder + '/predictions_' + args.eval_type +  "_" + args.domain + '/'
            visualfolder_savepath = args.results_savepath + result_folder + '/visuals_' + args.eval_type + "_" + args.domain + '/'
        else:
            resultfolder_path = args.results_savepath + result_folder + '/predictions_' + args.eval_type + '/'
            visualfolder_savepath = args.results_savepath + result_folder + '/visuals_' + args.eval_type + '/'
        os.makedirs(visualfolder_savepath, exist_ok=True)
        results_files = os.listdir(resultfolder_path)
        bar =  Bar('Processing predictions...', max = len(results_files))
        for reference in results_files:
            predictionfile_path = resultfolder_path + reference
            if os.path.isfile(predictionfile_path):
                file_savepath = visualfolder_savepath + reference[:-4] + '.jpg'
                predictions = np.load(predictionfile_path)
                Visualize_Predictions(predictions, file_savepath, args.dataset_name, args.background)
            bar.next()
        bar.finish()
        print('Predicted samples processing finished sucessfully')

                
    

if __name__ == '__main__':
    main()