import argparse
from options.baseoptions import BaseOptions
from options.dinooptions import DinoOptions
from options.deeplaboptions import DeepLabOptions
from options.deforestationoptions import DeforestationOptions
class TestOptions():
    def __init__(self):
        super(TestOptions, self).__init__()
        self.parser = argparse.ArgumentParser(description='')
    def initialize(self):
        self.parser = BaseOptions(self.parser).initialize()
        self.parser = DinoOptions(self.parser).initialize()
        self.parser = DeepLabOptions(self.parser).initialize()
        self.parser = DeforestationOptions(self.parser).initialize()
        self.parser.add_argument("--eval_type", default = "sliding_windows", type = str, help = "Evaluation type center_crop/resize/sliding_windows")
        self.parser.add_argument("--loader_crop", default = "center", type = str, help = "Evaluation type")
        self.parser.add_argument("--overlap_porcent", default = 0.05, type = float, help = "Overlap porcen of sliding windows")
        self.parser.add_argument("--procedure", default = "non_overlapped_center", type = str, help = "Procedure to form the mosaic average_overlap/non_overlapped_center")
        self.parser.add_argument('--batch_size_per_gpu', default=1, type=int, help='Per-GPU batch-size')
        self.parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
        self.parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
        self.parser.add_argument('--phase', default='get_metrics', help='evaluate model on validation set')
        return self.parser.parse_args()