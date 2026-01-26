import os
import argparse

class DeforestationOptions():
    def __init__(self, parser):
        super(DeforestationOptions, self).__init__()
        self.parser = parser
    
    def initialize(self):
        self.parser.add_argument('--task', default = 'deforestation_classifier', type = str, help='deforestation_classifier')
        self.parser.add_argument('--domains', type = str, nargs='+', default = ['RO'], help = 'Defines the domain will be used in training or evaluation')
        self.parser.add_argument("--porcent_pos_current_ref", type = int, default = 2, help = "Defins the percentage of pixel in the patch with deforested pixels")
        return self.parser