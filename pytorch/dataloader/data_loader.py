
import torch
import torchvision.transforms as transforms
from ipdb import set_trace as st


def CreateDataLoader(opt):
    data_transform = transforms.Compose([
        transforms.ToTensor(),  # divides float version by 255
        # normalize
    ])
    from dataloader.dataset import DatasetFromFolder

    val_loader = None

    if opt.test or opt.visualize:
        opt.batchSize = 1
        shuffle = False
        opt.data_augmentation = ["F", "F", "F", "F", "F"]
        split = opt.test_split
        phase = 'test'
    else:
        shuffle = True
        split = opt.train_split
        phase = 'train'

    crop = opt.use_crop
    resize = opt.use_resize
    
    if opt.validate:
        # if crop, get imagesize and define padding for tests
        set_valloader = DatasetFromFolder(opt, opt.dataroot, phase='val', data_split=opt.val_split, data_augmentation=["F", "F", "F", "F", "F"], crop=False, resize=resize, data_transform=data_transform, imageSize=opt.imageSize,
                                          outputSize=opt.outputSize, dataset_name=opt.dataset_name)
        val_loader = torch.utils.data.DataLoader(set_valloader, batch_size=1,
                                                 shuffle=False, num_workers=opt.nThreads)
    else:
        val_loader = None

    set_dataloader = DatasetFromFolder(opt, opt.dataroot, phase=phase, data_split=split, data_augmentation=opt.data_augmentation, crop=crop, resize=resize, data_transform=data_transform,
                                       imageSize=opt.imageSize, outputSize=opt.outputSize, dataset_name=opt.dataset_name)
    data_loader = torch.utils.data.DataLoader(
        set_dataloader, batch_size=opt.batchSize, shuffle=shuffle, num_workers=opt.nThreads)

    return data_loader, val_loader
