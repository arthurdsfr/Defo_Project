import argparse
from options.baseoptions import BaseOptions
from options.dinooptions import DinoOptions
from options.deeplaboptions import DeepLabOptions
from options.deforestationoptions import DeforestationOptions
class TrainOptions():
    def __init__(self):
        super(TrainOptions, self).__init__()
        self.parser = argparse.ArgumentParser(description='')
    def initialize(self):
        self.parser = BaseOptions(self.parser).initialize()
        self.parser = DinoOptions(self.parser).initialize()
        self.parser = DeepLabOptions(self.parser).initialize()
        self.parser = DeforestationOptions(self.parser).initialize()
        self.parser.add_argument("--train_type", default = "center_crop", type = str, help = "samples extraction type center_crop/resize/sliding_windows")
        self.parser.add_argument("--overlap_porcent", default = 0.90, type = float, help = "Overlap porcen of sliding windows")
        self.parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
        self.parser.add_argument("--lr", default=0.001, type=float, help="Learning rate at the beginning oftraining (highest LR used during training). The learning rate is linearly scaled with the batch size, and specified here for a reference batch size of 256. We recommend tweaking the LR depending on the checkpoint evaluated.""")
        self.parser.add_argument("--optimizer", default = "adam", type = str, help="Number of times the algorithm will be executed")
        self.parser.add_argument("--scheduler_type", default = "cosine", type = str, help = 'Defines the learning rate scheduler step')
        self.parser.add_argument("--weights", default = "None", type = str, help = "Defines the type of weights will be used in the cost function, manual set, None, and automatic computed")
        self.parser.add_argument("--cost_function", default="weighted_ce", type = str, help = "Defines which type of cost function will be used")
        self.parser.add_argument("--runs", default = 1, type = int, help="Number of times the algorithm will be executed")
        self.parser.add_argument('--batch_size_per_gpu', default=32, type=int, help='Per-GPU batch-size')
        self.parser.add_argument('--training_iterations', default = 0, type = int, help = "Maximum number of training iteratiosn per epoch")
        self.parser.add_argument('--validati_iterations', default = 0, type = int, help = "Maximum number of training iteratiosn per epoch")
        self.parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
        self.parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
        self.parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
        self.parser.add_argument('--patience', default=20, type=int, help="Epoch frequency for validation.")
        self.parser.add_argument('--phase', default='train', help='evaluate model on validation set')
        self.parser.add_argument("--training_graphs", type = eval, choices = [True, False], default = True, help = "Defines if graphs with the training behaviour will be saved at the end of the training")
        return self.parser.parse_args()