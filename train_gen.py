import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import torch.nn as nn
import torch
import torch.nn.functional as F
import pickle
import vtk
from vtk_utils.vtk_utils import *
import dataset
from torch.utils.data import DataLoader
import yaml
import functools
from gen_network import SDF4CHD, Tester
import pkbar
import matplotlib.pyplot as plt
from io_utils import plot_loss_curves, save_ckp, write_sampled_point, load_ckp
import io_utils
import argparse
import h5py
import random
import math
from torchinfo import summary
from network import act
from dataset import sample_points_from_sdf

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
print("DEVICE: ", device)

def write_sampled_point(points, point_values, fn):
    pts_py = (np.flip(np.squeeze(points[0].detach().cpu().numpy()), -1)+1.)/2.
    #pts_py = np.squeeze(points[0].detach().cpu().numpy())
    v_py = np.squeeze(point_values[0].detach().cpu().numpy())
    print(pts_py.shape, v_py.shape)
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(pts_py))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)

    v_arr = numpy_to_vtk(v_py.transpose())
    v_arr.SetName('occupancy')
    poly.GetPointData().AddArray(v_arr)
    verts = vtk.vtkVertexGlyphFilter()
    verts.AddInputData(poly)
    verts.Update()
    poly_v = verts.GetOutput()

    write_vtk_polydata(poly_v, fn)

def loss_func(sampled_gt_sdv, chd_type, type_z, type_s, outputs, kbar, i):
   
    recons_noDs_loss = torch.mean(((outputs['recons'][0].squeeze(-1) - sampled_gt_sdv)**2)*(sampled_gt_sdv+1))
    recons_loss = torch.mean(((outputs['recons'][1].squeeze(-1) - sampled_gt_sdv)**2)*(sampled_gt_sdv+1))
    
    gaussian_t_loss = torch.mean(type_z**2)
    gaussian_s_loss = torch.mean(type_s**2)

    total_loss = 1. * (recons_loss) + 1.5 * recons_noDs_loss + \
           0.001*gaussian_t_loss + 0.0001*gaussian_s_loss

    kbar.update(i, values=[("loss", total_loss), ("recons", recons_loss), ("recons_noDs", recons_noDs_loss),  
                    ("gaussian_s_loss", gaussian_s_loss), ("gaussian_t_loss", gaussian_t_loss)])

    return total_loss

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

    with open(args.config, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)

    if not os.path.exists(cfg['data']['output_dir']):
        os.makedirs(cfg['data']['output_dir'])

    # create dataloader
    if cfg['net']['encoder']:
        train = dataset.ImgSDFDataset(cfg['data']['train_dir'], cfg['train']['n_smpl_pts'], cfg['data']['point_sampling_factor'], \
            cfg['data']['chd_info'], mode=['aligned_train'], use_aug=True)
    else:
        train = dataset.SDFDataset(cfg['data']['train_dir'], cfg['train']['n_smpl_pts'], cfg['data']['point_sampling_factor'], \
            cfg['data']['chd_info'], mode=['aligned_train'], use_aug=True)
    dataloader_train = DataLoader(train, batch_size=cfg['train']['batch_size'], shuffle=True, pin_memory=True, drop_last=True, worker_init_fn = worker_init_fn, num_workers=1)

    # create network and latent codes
    if cfg['net']['encoder']:
        net = SDF4CHD(in_dim=cfg['net']['encoder_in_dim'], \
                out_dim=cfg['net']['out_dim'], \
                num_types=len(cfg['data']['chd_info']['types']), \
                z_t_dim=cfg['net']['z_t_dim'], \
                z_s_dim=cfg['net']['z_s_dim'], \
                type_mlp_num=cfg['net']['type_mlp_num'],\
                ds_mlp_num=cfg['net']['ds_mlp_num'],\
                dx_mlp_num=cfg['net']['dx_mlp_num'], \
                latent_dim=cfg['net']['latent_dim'])
    else:
        net = SDF4CHD(in_dim=0, \
                out_dim=cfg['net']['out_dim'], \
                num_types=len(cfg['data']['chd_info']['types']), \
                z_t_dim=cfg['net']['z_t_dim'], \
                z_s_dim=cfg['net']['z_s_dim'], \
                type_mlp_num=cfg['net']['type_mlp_num'],\
                ds_mlp_num=cfg['net']['ds_mlp_num'],\
                dx_mlp_num=cfg['net']['dx_mlp_num'], \
                latent_dim=cfg['net']['latent_dim'])
        # initialize Z_s
        lat_vecs = torch.nn.Embedding(len(train), cfg['net']['z_s_dim']*2*2*2, max_norm=50.).to(device)
        torch.nn.init.kaiming_normal_(lat_vecs.weight.data, a=0.02, nonlinearity='leaky_relu')
    net.to(device)
    
    # no weight decay for type prediction
    subnets_nodecay = [net.type_encoder, net.decoder.decoder]
    params_nodecay = set()
    for n in subnets_nodecay:
        params_nodecay |= set(n.parameters())
    params_decay = set(net.parameters()).difference(params_nodecay)

    optimizer_nodecay = torch.optim.Adam(params_nodecay, lr=cfg['train']['lr'], betas=(0.5, 0.999), weight_decay=0.0)
    optimizer_decay = torch.optim.Adam(params_decay, lr=cfg['train']['lr'], betas=(0.5, 0.999), weight_decay=0.0001)
    scheduler_nodecay = torch.optim.lr_scheduler.StepLR(optimizer_nodecay, step_size=cfg['train']['scheduler']['patience'], gamma=cfg['train']['scheduler']['factor'])
    scheduler_decay = torch.optim.lr_scheduler.StepLR(optimizer_decay, step_size=cfg['train']['scheduler']['patience'], gamma=cfg['train']['scheduler']['factor'])
    if not cfg['net']['encoder']:
        optimizer_zs = torch.optim.Adam(lat_vecs.parameters(), lr=0.01, betas=(0.5, 0.999))  
        scheduler_zs = torch.optim.lr_scheduler.StepLR(optimizer_zs, step_size=cfg['train']['latent_scheduler']['patience'], gamma=cfg['train']['latent_scheduler']['factor'])

    # start training
    for epoch in range(start_epoch, cfg['train']['epoch']):
        kbar = pkbar.Kbar(target=len(dataloader_train), epoch=epoch, num_epochs=cfg['train']['epoch'], width=20, always_stateful=False)
        net.train()
        for i, data in enumerate(dataloader_train):
            if cfg['net']['encoder']:
                z_s = data['image'].to(device)
            else:
                z_s = lat_vecs(data['idx'].to(device))
                z_s = z_s.view(cfg['train']['batch_size'], cfg['net']['z_s_dim'], 2, 2, 2)
            points = data['points'].to(device)
            point_values = data['pt_sdv'].to(device)
            chd_type = data['chd_type'].to(device)
            if epoch == start_epoch and i ==0:
                summary(net, [tuple(z_s.shape), tuple(points.shape), tuple(chd_type.shape)])
                #write_sampled_point(points, point_values, os.path.join(cfg['data']['output_dir'], 'sample_{}_epoch{}.vtp'.format(i, epoch)))
            net.zero_grad()
            outputs, z_t = net(z_s, points, chd_type) 
            loss = loss_func(point_values, chd_type, z_t, z_s, outputs, kbar, i)
            loss.backward()
            optimizer_nodecay.step()
            optimizer_decay.step()
            if not cfg['net']['encoder']:
                optimizer_zs.step()
        with torch.no_grad():
            scheduler_nodecay.step()
            scheduler_decay.step()
            if not cfg['net']['encoder']:
                scheduler_zs.step()
            if (epoch+1) % 1 ==0:
                if cfg['net']['encoder']:
                    optimizer_list = [optimizer_nodecay, optimizer_decay]
                else:
                    optimizer_list = [optimizer_nodecay, optimizer_decay, optimizer_zs]
                    all_latents = lat_vecs.state_dict()
                    torch.save({'epoch': epoch+1, 'latent_codes': all_latents}, os.path.join(cfg['data']['output_dir'], 'code{}.pt'.format(epoch+1)))
                save_ckp(net, optimizer_list, epoch, False, cfg['data']['output_dir'], cfg['data']['output_dir']) 
