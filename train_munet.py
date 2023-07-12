import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import torch.nn as nn
import torch
import torch.nn.functional as F
import pickle
import vtk
from vtk_utils.vtk_utils import *
import munet_dataset
from torch.utils.data import DataLoader
import yaml
import functools
import pkbar
import matplotlib.pyplot as plt
import io_utils
import argparse
import h5py
import random
from torchinfo import summary
from munet import Modified3DUNet

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
print("DEVICE: ", device)
def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config') 
    args = parser.parse_args()
    start_epoch = 0

    config_fn = args.config
    with open(config_fn, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)

    if not os.path.exists(cfg['data']['output_dir']):
        os.makedirs(cfg['data']['output_dir'])

    # create dataloader
    #train = dataset.MeshDataset(cfg['data']['train_dir'], cfg['train']['n_smpl_pts'])
    train = munet_dataset.ImgSDFDataset(cfg['data']['train_dir'], cfg['data']['chd_info'], mode=['train'], use_aug=True)
    test = munet_dataset.ImgSDFDataset(cfg['data']['test_dir'], cfg['data']['chd_info'], mode=['validate'], use_aug=False)
    dataloader_train = DataLoader(train, batch_size=2, shuffle=True, pin_memory=True, drop_last=True, worker_init_fn = worker_init_fn, num_workers=4)
    dataloader_test = DataLoader(test, batch_size=2, shuffle=True, pin_memory=True, drop_last=True, worker_init_fn = worker_init_fn, num_workers=4)
    print("Steps: ", len(dataloader_train), len(dataloader_test))
   
    #net = Modified3DUNet(in_channels=1, n_classes=8, base_n_filter=16) 
    #net = Modified3DUNet(in_channels=1, n_classes=7, base_n_filter=16) 
    net = Modified3DUNet(in_channels=1, n_classes=1, base_n_filter=16) 
    net = nn.DataParallel(net)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['train']['lr'], betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg['train']['scheduler']['factor'],
                                                           patience=cfg['train']['scheduler']['patience'], min_lr=1e-6)
    if os.path.exists(os.path.join(cfg['data']['output_dir'], 'net.pt')):
        print("LOADING LASTEST CHECKPOINT")
        net, optimizer, scheduler, start_epoch = io_utils.load_ckp_single(os.path.join(cfg['data']['output_dir'], 'net.pt'), net, optimizer, scheduler)

    best_val_loss = float('inf')
    #loss_func = torch.nn.CrossEntropyLoss()
    #l1 = torch.nn.L1Loss(reduction="mean")
    l2 = torch.nn.MSELoss(reduction="none")
    l2 = torch.nn.BCELoss(reduction="none")
    for epoch in range(start_epoch, cfg['train']['epoch']-start_epoch):
        kbar = pkbar.Kbar(target=len(dataloader_train), epoch=epoch, num_epochs=cfg['train']['epoch'], width=20, always_stateful=False)
        net.train()
        for i, data in enumerate(dataloader_train):
            img = data['image'].to(device)
            #if epoch == 0 and i ==0:
            #    summary(net, tuple(img.shape)) 
            net.zero_grad()
            #gt  = data['y'].long().to(device)
            #logits = net(img)
            #loss = loss_func(logits, gt)
            sdf_gt = data['processed_sdf'].to(device)
            sdf, _ = net(img)
            loss = torch.mean(l2(sdf, sdf_gt) * (sdf_gt + 1)) 
            
            loss.backward()
            optimizer.step()
            kbar.update(i, values=[("loss", loss)])
        with torch.no_grad():
            net.eval()
            total_loss = 0.
            for i, data in enumerate(dataloader_test):
                img = data['image'].to(device)
                #gt  = data['y'].long().to(device)
                #logits = net(img)
                #val_loss = loss_func(logits, gt)
                sdf, _ = net(img)
                sdf_gt = data['processed_sdf'].to(device)
                #val_loss = l1(torch.clamp(sdf, min=-0.5, max=0.5), torch.clamp(sdf_gt, min=-0.5, max=0.5))
                val_loss = torch.mean(l2(sdf, sdf_gt) * (sdf_gt + 1)) 
                total_loss += val_loss.item()
            scheduler.step(total_loss)
            kbar.add(1, values=[("val_total_loss", total_loss/len(dataloader_test))])
            best = total_loss < best_val_loss
            if best:
                best_val_loss = total_loss
            io_utils.save_ckp_single(net, optimizer, scheduler, epoch, os.path.join(cfg['data']['output_dir']))
            
