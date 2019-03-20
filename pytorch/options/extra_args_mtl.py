from .arguments import TrainOptions

class MTL_Options(TrainOptions):
    def initialize(self):
        TrainOptions.initialize(self)
        self.parser.add_argument('--mtl_method', default='eweights')
        self.parser.add_argument('--tasks', nargs='+', default=['depth'], help='all tasks separated by a space')
        self.parser.add_argument('--outputs_nc', nargs='+', default=[1], type=int, help='all tasks separated by a space')
        self.parser.add_argument('--regression_loss', default='L1')
        self.parser.add_argument('--alpha', type=float, default=0.5, help='weight of losses for semantic and depth')
        