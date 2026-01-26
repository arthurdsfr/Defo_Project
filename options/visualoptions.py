import argparse
from options.baseoptions import BaseOptions

class VisualOptions():
    def __init__(self):
        super(VisualOptions, self).__init__()
        self.parser = argparse.ArgumentParser(description='')
    
    def initialize(self):
        #self.parser = BaseOptions(self.parser).initialize()
        self.parser.add_argument('--phase', default='get_visuals', help='evaluate model on validation set')
        return self.parser.parse_args()