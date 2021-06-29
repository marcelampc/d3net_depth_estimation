# simplified main
from options.extra_args_mtl import MTL_Options as TrainOptions
from dataloader.data_loader import CreateDataLoader

# Load options
opt = TrainOptions().parse()

# train model
if opt.train or opt.resume:
    from models.mtl_train import MultiTaskGen as Model
    model = Model()
    model.initialize(opt)
    data_loader, val_loader = CreateDataLoader(opt)
    model.train(data_loader, val_loader=val_loader)
elif opt.test:
    from models.mtl_test import MTL_Test as Model
    model = Model()
    model.initialize(opt)
    model.test()