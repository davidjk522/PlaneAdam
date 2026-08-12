import re
import os
import glob
import torch
import numpy as np

from torch.utils.data import DataLoader

from models import getModel
from loaders.acdcreg_loader import acdcreg_loader
from loaders.abdomenreg_loader import abdomenreg_loader
from loaders.lungreg_loader import lungreg_loader
from loaders.oasis_pkl_loader import oasis_pkl_loader
from loaders.oasis_2d_loader import oasis_2d_loader 
from loaders.ixi_pkl_loader import ixi_pkl_loader
from loaders.thoraxreg_loader import thoraxreg_loader
from loaders.thoraxregvalonly_loader import thoraxregvalonly_loader
from loaders.abdomenorireg_loader import abdomenorireg_loader
from loaders.mmreg_loader import mmreg_loader
from utils.functions import modelSaver, convert_state_dict, dice_binary

def loadDataset(opt, split = 'train'):

    dataset_name = opt['dataset']
    data_path = opt['data_path']

    if dataset_name == 'acdcreg':
        loader = acdcreg_loader(root_dir = data_path, split = split)
    elif dataset_name == 'lungreg':
        loader = lungreg_loader(root_dir = data_path, split = split)
    elif dataset_name == 'mmreg':
        loader = mmreg_loader(root_dir = data_path, split = split)
    elif dataset_name == 'oasisreg':
        loader = oasis_pkl_loader(root_dir = data_path, split = split)
    elif dataset_name == 'oasis_2d':
        loader = oasis_2d_loader(root_dir = data_path, split = split)
    elif dataset_name == 'ixireg':
        loader = ixi_pkl_loader(root_dir = data_path, split = split)
    elif dataset_name == 'abdomenreg':
        loader = abdomenreg_loader(root_dir = data_path, split = split)
    elif dataset_name == 'thoraxreg':
        loader = thoraxreg_loader(root_dir = data_path, split = split)
    elif dataset_name == 'thoraxregvalonly':
        loader = thoraxregvalonly_loader(root_dir = data_path, split = split)
    elif dataset_name == 'abdomenorireg':
        loader = abdomenorireg_loader(root_dir = data_path, split = split)
    else:
        raise ValueError('Unkown datasets: please define proper dataset name')

    print("----->>>> %s dataset is loaded ..." % dataset_name)

    return loader

def getDataLoader(opt, split='train'):

    if split == 'train':
        data_shuffle = True
        batch_size = opt['batch_size']
    else:
        data_shuffle = False
        batch_size = 1

    num_workers = opt['num_workers']
    print("----->>>> Loading %s dataset ..." % (split))
    dataset = loadDataset(opt, split)
    loader = DataLoader(dataset = dataset,
                        num_workers = num_workers,
                        batch_size = batch_size,
                        pin_memory = True,
                        shuffle = data_shuffle)
    print("----->>>> %s batch size: %d, # of %s iterations per epoch: %d" %  (split, batch_size, split, int(len(dataset) / batch_size)))

    return loader

def getModelSaver(opt, suffix=None):

    if suffix is None:
        model_saver = modelSaver(opt['log'], opt['save_freq'], opt['n_checkpoints'])
    else:
        sv_path = os.path.join(opt['log'], suffix)
        os.makedirs(sv_path, exist_ok=True)
        model_saver = modelSaver(sv_path, opt['save_freq'], opt['n_checkpoints'])

    return model_saver

def findLastCheckpoint(save_path):

    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall("net_epoch_(.*)_score_.*.pth.*", file_)
            if result:
                epochs_exist.append(int(result[0]))
        init_epoch = max(epochs_exist)
    else:
        init_epoch = 0

    score = None
    if init_epoch > 0:
        for file_ in file_list:
            file_name = "net_epoch_" + str(init_epoch) + "_score_(.*).pth.*"
            result = re.findall(file_name, file_)
            if result:
                score = result[0]
                break

    return_name = None
    if init_epoch > 0:
        return_name =  "net_epoch_" + str(init_epoch) + "_score_" + score + ".pth"

    return init_epoch, score, return_name

def findBestCheckpoint(save_path):

    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        scores = []
        for file_ in file_list:
            result = re.findall("best_score_(.*)_net_epoch_.*.pth.*", file_)
            if result:
                epochs_exist.append(result[0])
                scores.append(float(result[0]))
        ind = np.argmax(scores)
        score = epochs_exist[ind]
        for file_ in file_list:
            file_name = "best_score_" + str(score) + "_net_epoch_.*.pth.*"
            result = re.findall(file_name, file_)
            if result:
                return_name = result[0]
                file_name = "best_score_" + str(score) + "_net_epoch_(.*).pth.*"
                result = re.findall(file_name, file_)
                epoch = result[0]
                return epoch, score, return_name

    raise ValueError("can't find checkpoints")

def findCheckpointByEpoch(save_path, epoch):

    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        for file_ in file_list:
            file_name = "net_epoch_" + str(epoch) + "_score_.*.pth.*"
            result = re.findall(file_name, file_)
            if result:
                return result[0]

    raise ValueError("can't find checkpoints")

def findBestDiceByEpoch(save_path, epoch):

    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        for file_ in file_list:
            file_name = "best_score_.*_net_epoch_" + str(epoch) + ".pth.*"
            result = re.findall(file_name, file_)
            if result:
                return result[0]

    raise ValueError("can't find checkpoints")

def getTrainModelWithCheckpoints(opt, model_type=None):

    print("----->>>> Loading model %s " % model_type)
    
    init_epoch = 0
    model = getModel(opt)

    if model_type is None:
        return model, init_epoch
    
    print("----->>>> Loading model from %s " % opt['log'])
    if model_type == 'last':
        init_epoch, score, file_name = findLastCheckpoint(opt['log'])
    elif model_type == 'best':
        init_epoch, score, file_name = findBestCheckpoint(opt['log'])
    else:
        if 'best' in model_type:
            st = model_type.split('_')[-1]
            opt['log'] = os.path.join(opt['log'], st)
            init_epoch, score, file_name = findBestCheckpoint(opt['log'])
    init_epoch = int(init_epoch)
    if init_epoch > 0:
        print("----->>>> Resuming model by loading epoch %s with dice %s" % (init_epoch, score))
        states = convert_state_dict(torch.load(os.path.join(opt['log'], file_name)))
        model.load_state_dict(states)

    return model, init_epoch

def getTestModelWithCheckpoints(opt):

    model = getModel(opt)
    file_name = 'unknown'
    epoch = '0'
    score = '0'
    which_model = 'unknown'
    states = None
    #print('tttttt',opt['log'])

    if opt['load_ckpt'] == 'best':
        epoch, score, file_name = findBestCheckpoint(opt['log'])
        which_model = 'best'
    elif 'best' in opt['load_ckpt']:
        st = opt['load_ckpt'].split('_')[-1]
        opt['log'] = os.path.join(opt['log'], st)
        epoch, score, file_name = findBestCheckpoint(opt['log'])
        which_model = 'best'
    elif opt['load_ckpt'] == 'last':
        epoch, score, file_name = findLastCheckpoint(opt['log'])
        which_model = 'last'
    elif 'last' in opt['load_ckpt']:
        st = opt['load_ckpt'].split('_')[-1]
        opt['log'] = os.path.join(opt['log'], st)
        epoch, score, file_name = findLastCheckpoint(opt['log'])
        which_model = 'last'
    elif "epoch" in opt['load_ckpt']:
        epoch = opt['load_ckpt'].split('_')[1]
        file_name = findCheckpointByEpoch(opt['log'], epoch)
        which_model = str(epoch) + 'th'
    elif opt['load_ckpt'] == 'none':
        print("----->>>> No model is loaded")
    elif os.path.exists(opt['load_ckpt']):
        print("----->>>> Loading model from %s" % opt['load_ckpt'])
        epoch, score = '-1', '-1'
        states = convert_state_dict(torch.load(opt['load_ckpt']))
        model.load_state_dict(states)
    else:
        raise ValueError("Not either best, last, epoch or none, or a valid path to a checkpoint")

    if file_name != 'unknown':
        print("----->>>> Resuming the %s model by loading epoch %s with dice %s" % (which_model, epoch, score))
        states = convert_state_dict(torch.load(os.path.join(opt['log'], file_name)))
        model.load_state_dict(states)

    info = {
        "file_name": file_name,
        "epoch": int(epoch),
        "score": float(score),
    }

    return model, states #info
