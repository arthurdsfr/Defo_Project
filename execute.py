import os
import sys
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--running_in', type = str, default = 'localpc', help = 'Ste the environment where the experiment will run')
parser.add_argument('--tasks', type=str, default = 'getmetrics',help = 'specifies which operation will be executed')
parser.add_argument('--experiment_names', type = str, default='DEFORESTATION_DETECTION_P', help = 'Defines the name of the experiment')

args = parser.parse_args()
print(args)
Schedule = []

if args.running_in == 'localpc':
    Train_MAIN_COMMAND = "./Deforestation_Detection/train.py"
    Test_MAIN_COMMAND = "./Deforestation_Detection/test.py"
    GetMetrics_MAIN_COMMAND = "./Deforestation_Detection/get_metrics.py"
    GetVisuals_MAIN_COMMAND = "./Deforestation_Detection/get_visuals.py"
    EXPERIMENT_MAINPATH = "./EXPERIMENTS/"
    CSVFILE_PATH = "None"
    DATA_PATH = "./DATA/"


#Base options
FEATUREEXTRACTOR_ARCH = 'vnet'
SEGMENTATIONHEAD_ARCH = 'None'
INPUT_PATCH_SIZE = '128'
DATASET_NAME = 'deforestation'
BACKGROUND = 'True'
NUM_CLASSES = '2'
MULTIPLE_GPUS = "False"
FE_PRETRAINED_WEIGHTS = "None"
SH_PRETRAINED_WEIGHTS = "None"
OVERAL_PROJECTNAME = '/PROVE'
EXPERIMENT_NAME = '/' + args.experiment_names + '/'
# Deforestation options
TASK = "deforestation_classifier"
DOMAINS = "MA"
# Feature extractor options, wiil be set depending the architecture we choose
# DeepLab options
DEEPLABOUTPUT_STRIDE = '16'
DEEPLABBACKBONE = 'resnet'
# Dino options
DINO_PATCH_SIZE = '16'
DINO_CHECKPOINT_KEY = 'teacher'
DINO_ARCH = 'vit_base'
IMAGENET_PRETRAIN = 'True'
FREEZE_FEATUREEXTRACTOR = 'False'
# Train options
TRAIN_TYPE = 'sliding_windows'
TRAIN_OVERLAP_PORCENT = '0.95'
EPOCHS = '5'
LEARNING_RATE = '0.0001'
OPTIMIZER = 'adam'
WEIGHTS = 'manual'
SCHEDULER_TYPE = "cosine"
COST_FUNCTION = "cross_entropy"
RUNS = '1'
TRAIN_BATCH_SIZE = '32'
TRAINING_ITERATIONS = '0'
VALIDITI_ITERATIONS = '0'
DIST_URL = 'env://'
VAL_FREQ = '1'
PATIENCE = '20'
NUM_WORKERS = '6'
TRAINING_GRAPHS = 'True'
# Test options
TEST_TYPE = "sliding_windows"
CROP_TYPE = "None"
TEST_OVERLAP_PORCENT = "0.55"
PROCEDURE = "non_overlapped_center"
TEST_BATCH_SIZE = "1"
if "train" in args.tasks:
    Schedule.append("python " + Train_MAIN_COMMAND + " "
                    "--featureextractor_arch " + FEATUREEXTRACTOR_ARCH + " "
                    "--segmentationhead_arch " + SEGMENTATIONHEAD_ARCH + " "
                    "--input_patch_size " + INPUT_PATCH_SIZE + " "
                    "--dataset_name " + DATASET_NAME + " "
                    "--background " + BACKGROUND + " "
                    "--num_classes " + NUM_CLASSES + " "
                    "--multiple_gpus " + MULTIPLE_GPUS + " "
                    "--fe_pretrained_weights " + FE_PRETRAINED_WEIGHTS + " "
                    "--sh_pretrained_weights " + SH_PRETRAINED_WEIGHTS + " "
                    "--data_path " + DATA_PATH + " "
                    "--csv_path " + CSVFILE_PATH + " "
                    "--experiment_mainpath " + EXPERIMENT_MAINPATH + " "
                    "--overall_projectname " + OVERAL_PROJECTNAME + " "
                    "--experiment_name " + EXPERIMENT_NAME + " "
                    "--task " + TASK + " "
                    "--domains " + DOMAINS + " "
                    "--deeplaboutput_stride " + DEEPLABOUTPUT_STRIDE + " "
                    "--deeplabbackbone " + DEEPLABBACKBONE + " "
                    "--dino_patch_size " + DINO_PATCH_SIZE + " "
                    "--dino_checkpoint_key " + DINO_CHECKPOINT_KEY + " "
                    "--dino_arch " + DINO_ARCH + " "
                    "--dino_imagenet_pretrain " + IMAGENET_PRETRAIN + " "
                    "--freeze_featureextractor " + FREEZE_FEATUREEXTRACTOR + " "
                    "--train_type " + TRAIN_TYPE + " "
                    "--overlap_porcent " + TRAIN_OVERLAP_PORCENT + " "
                    "--epochs " + EPOCHS + " "
                    "--lr " + LEARNING_RATE + " "
                    "--optimizer " + OPTIMIZER + " "
                    "--scheduler_type " + SCHEDULER_TYPE + " "
                    "--weights " + WEIGHTS + " "
                    "--cost_function " + COST_FUNCTION + " "
                    "--runs " + RUNS + " "
                    "--batch_size_per_gpu " + TRAIN_BATCH_SIZE + " "
                    "--training_iterations " + TRAINING_ITERATIONS + " "
                    "--validati_iterations " + VALIDITI_ITERATIONS + " "
                    "--dist_url " + DIST_URL + " "
                    "--val_freq " + VAL_FREQ + " "
                    "--patience " + PATIENCE + " "
                    "--num_workers " + NUM_WORKERS + " "
                    "--phase train "
                    "--training_graphs " + TRAINING_GRAPHS + ""
    )
if "test" in args.tasks:
    TASK = "deforestation_classifier"
    DOMAINS = "MA"
    Schedule.append("python " + Test_MAIN_COMMAND + " "
                    "--featureextractor_arch " + FEATUREEXTRACTOR_ARCH + " "
                    "--segmentationhead_arch " + SEGMENTATIONHEAD_ARCH + " "
                    "--input_patch_size " + INPUT_PATCH_SIZE + " "
                    "--dataset_name " + DATASET_NAME + " "
                    "--background " + BACKGROUND + " "
                    "--num_classes " + NUM_CLASSES + " "
                    "--multiple_gpus " + MULTIPLE_GPUS + " "
                    "--fe_pretrained_weights " + FE_PRETRAINED_WEIGHTS + " "
                    "--sh_pretrained_weights " + SH_PRETRAINED_WEIGHTS + " "
                    "--data_path " + DATA_PATH + " "
                    "--csv_path " + CSVFILE_PATH + " "
                    "--experiment_mainpath " + EXPERIMENT_MAINPATH + " "
                    "--overall_projectname " + OVERAL_PROJECTNAME + " "
                    "--experiment_name " + EXPERIMENT_NAME + " "
                    "--task " + TASK + " "
                    "--domains " + DOMAINS + " "
                    "--deeplaboutput_stride " + DEEPLABOUTPUT_STRIDE + " "
                    "--deeplabbackbone " + DEEPLABBACKBONE + " "
                    "--dino_patch_size " + DINO_PATCH_SIZE + " "
                    "--dino_checkpoint_key " + DINO_CHECKPOINT_KEY + " "
                    "--dino_arch " + DINO_ARCH + " "
                    "--dino_imagenet_pretrain " + IMAGENET_PRETRAIN + " "
                    "--freeze_featureextractor " + FREEZE_FEATUREEXTRACTOR + " "
                    "--eval_type " + TEST_TYPE + " " 
                    "--loader_crop " + CROP_TYPE + " "
                    "--overlap_porcent " + TEST_OVERLAP_PORCENT + " "
                    "--procedure " + PROCEDURE + " "
                    "--batch_size_per_gpu " + TEST_BATCH_SIZE + " "
                    "--dist_url " + DIST_URL + " "
                    "--phase test"
    )
if "getmetrics" in args.tasks:
    TASK = "deforestation_classifier"
    DOMAINS = "MA"
    Schedule.append("python " + GetMetrics_MAIN_COMMAND + " "
                    "--featureextractor_arch " + FEATUREEXTRACTOR_ARCH + " "
                    "--segmentationhead_arch " + SEGMENTATIONHEAD_ARCH + " "
                    "--input_patch_size " + INPUT_PATCH_SIZE + " "
                    "--dataset_name " + DATASET_NAME + " "
                    "--background " + BACKGROUND + " "
                    "--num_classes " + NUM_CLASSES + " "
                    "--multiple_gpus " + MULTIPLE_GPUS + " "
                    "--fe_pretrained_weights " + FE_PRETRAINED_WEIGHTS + " "
                    "--sh_pretrained_weights " + SH_PRETRAINED_WEIGHTS + " "
                    "--data_path " + DATA_PATH + " "
                    "--csv_path " + CSVFILE_PATH + " "
                    "--experiment_mainpath " + EXPERIMENT_MAINPATH + " "
                    "--overall_projectname " + OVERAL_PROJECTNAME + " "
                    "--experiment_name " + EXPERIMENT_NAME + " "
                    "--task " + TASK + " "
                    "--domains " + DOMAINS + " "
                    "--eval_type " + TEST_TYPE + " " 
                    "--loader_crop " + CROP_TYPE + " "
                    "--overlap_porcent " + TEST_OVERLAP_PORCENT + " "
                    "--procedure " + PROCEDURE + " "
                    "--batch_size_per_gpu " + TEST_BATCH_SIZE + ""
    )

for i in range(len(Schedule)):
    os.system(Schedule[i])
    