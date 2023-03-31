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
import io_utils
import metrics
import SimpleITK as sitk
import argparse
import glob
import time
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def fit_testdata(net, cfg, mode, iter_num=200, num_blocks=1, thresh=0.5, output_mode='sdf'):
    STD = 1.
    fns = sorted(glob.glob(os.path.join(cfg['data']['test_dir'], 'pytorch', '*.pkl')))
    print("TEST: ", fns, os.path.join(cfg['data']['test_dir'], 'pytorch', '*.pkl'))
    df = dataset.read_excel(cfg['data']['chd_info']['diag_fn'], sheet_name=cfg['data']['chd_info']['diag_sn'])
    fns, chd_data = dataset.parse_data_by_chd_type(fns, df, cfg['data']['chd_info']['types'], mode=mode, use_aug=False) 
    dice_list = []
    for i, fn in enumerate(fns):
        net.train()
        for param in net.parameters():
            param.requires_grad = False
        input_data = pickle.load(open(fn, 'rb'))
        seg_py = np.argmin(input_data, axis=0)+1
        seg_py[np.all(input_data>0., axis=0)] = 0
        filename = os.path.basename(fn).split('.')[0]
        
        z_s = torch.normal(torch.zeros((1, cfg['net']['z_s_dim'], 2, 2, 2)), std=STD).to(device)
        z_s.requires_grad=True
        
        chd_type = torch.from_numpy(chd_data[i, :].astype(np.float32)).unsqueeze(0).to(device)
        optimizer = torch.optim.Adam([z_s], lr=0.05, betas=(0.5, 0.999), weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
        z_t = net.type_encoder(chd_type) 

        for k in range(iter_num):
            optimizer.zero_grad()
            _, points, point_values = dataset.sample_points_from_sdf(input_data, 32768, 20)
            points = points.unsqueeze(0).to(device)
            point_values = point_values.unsqueeze(0).to(device)
            outputs = net.decoder(z_s, z_t, points)

            recons_loss = torch.mean(((outputs['recons'][0].squeeze(-1) - point_values)**2)*(point_values+1))
            recons_noDs_loss = torch.mean(((outputs['recons'][1].squeeze(-1) - point_values)**2)*(point_values+1))
            gaussian_s_loss = torch.mean(z_s**2)
            total_loss = 1.*recons_loss + 1.5*recons_noDs_loss + 0.0001 * gaussian_s_loss
            total_loss.backward()
            print(k, total_loss.item(), recons_loss.item(), recons_noDs_loss.item(), gaussian_s_loss.item())
            optimizer.step()
            scheduler.step()
        with torch.no_grad():
            if output_mode=='mesh':
                type_sdf = tester.z2voxel(z_s, z_t, net, num_blocks=num_block, out_type=True)
                out_type_fn = os.path.join(cfg['data']['test_output_dir'], '{}_test_type_struct.vtp'.format(filename))
                type_mesh = io_utils.write_sdf_to_vtk_mesh(type_sdf, out_type_fn, thresh, decimate=0.5)
                points = np.flip(vtk_to_numpy(type_mesh.GetPoints().GetData()), axis=-1)
                points = torch.from_numpy(points.copy()).unsqueeze(0).to(device) * 2. - 1.
                new_points, _ = net.decoder.flow_layer(points, z_t, z_s, inverse=False)
                new_points = (np.flip(new_points.detach().cpu().numpy(), -1) +1.)/2.
                type_mesh.GetPoints().SetData(numpy_to_vtk(np.squeeze(new_points)))
                write_vtk_polydata(type_mesh, os.path.join(cfg['data']['test_output_dir'], '{}_test_deformed_mesh.vtp'.format(filename)))
            else:
                sdf = tester.z2voxel(z_s, z_t, net, num_blocks=num_block, out_block=-1, get_tmplt_coords=False)
                sdf_noCorr = tester.z2voxel(z_s, z_t, net, num_blocks=num_block, out_block=0, get_tmplt_coords=False)
                io_utils.write_sdf_to_vtk_mesh(sdf, os.path.join(cfg['data']['test_output_dir'], '{}_test_pred.vtp'.format(filename)), thresh) 
                io_utils.write_sdf_to_vtk_mesh(sdf_noCorr, os.path.join(cfg['data']['test_output_dir'], '{}_test_pred_noCorr.vtp'.format(filename)), thresh) 
                dice = metrics.dice_score_from_sdf(sdf[1:-1, 1:-1, 1:-1], seg_py)
                print(dice)
                dice_list.append(dice)
    if len(dice_list)>0:
        io_utils.write_scores(os.path.join(cfg['data']['test_output_dir'], 'dice_test.csv'), dice_list)


def sample_shape_space(net, data, cfg, lat_vecs=None, num_copies=10, num_block=1, thresh=0.5, get_img=False):
    import random
    chd_type = data['chd_type'].to(device)
    filename = data['filename']
    z_t = net.type_encoder(chd_type)
    if get_img:
        if cfg['net']['encoder']:
            img = data['image'].to(device)
            z_s = net.encoder(img)
        else:
            z_s = lat_vecs(data['idx'].to(device)).view(1, cfg['net']['z_s_dim'], 2, 2, 2)
    dir_n = os.path.join(os.path.dirname(cfg['data']['train_output_dir']), 'aligned_train')
    for i in range(num_copies):
        dx_z_s = torch.normal(torch.zeros((1, cfg['net']['z_s_dim'], 2, 2, 2)), std=1.).to(chd_type.device)
        sdf = tester.z2voxel(dx_z_s, z_t, net, num_blocks=num_block, out_block=-1, get_tmplt_coords=False)
        sdf_noCorr = tester.z2voxel(dx_z_s, z_t, net, num_blocks=num_block, out_block=0, get_tmplt_coords=False)
        io_utils.write_sdf_to_vtk_mesh(sdf, os.path.join(cfg['data']['train_output_dir'], '{}_pred_r{}.vtp'.format(filename[0], i)), thresh) 
        io_utils.write_sdf_to_vtk_mesh(sdf_noCorr, os.path.join(cfg['data']['train_output_dir'], '{}_pred_noCorr_r{}.vtp'.format(filename[0], i)), thresh) 
        if get_img:
            gt_sdf = data['sdf'].to(device)
            new_img = tester.deform_image(z_s, dx_z_s, z_t, img, net, num_blocks=num_block, out_block=-1, order=1)
            new_sdf = tester.deform_image(z_s, dx_z_s, z_t, gt_sdf, net, num_blocks=num_block, out_block=-1, order=0)
            new_seg = np.argmin(new_sdf, axis=-1)+1
            new_seg[np.all(new_sdf>0., axis=-1)] = 0
            io_utils.write_nifty_image(new_seg.astype(np.uint8), os.path.join(cfg['data']['train_output_dir'], '{}_pred_seg_r{}.nii.gz'.format(filename[0], i)))
            io_utils.write_nifty_image(new_img, os.path.join(cfg['data']['train_output_dir'], '{}_pred_im_r{}.nii.gz'.format(filename[0], i)))

def sample_type_space(net, data, cfg, lat_vecs=None, num_block=1, thresh=0.5):
    if cfg['net']['encoder']:
        img = data['image'].to(device)
        z_s = net.encoder(img)
    else:
        z_s = lat_vecs(data['idx'].to(device)).view(1, cfg['net']['z_s_dim'], 2, 2, 2)
    # CA
    types = {}
    types['CA'] = [[0., 0., 0., 0., 1., 0.]]
    types['PuA']= [[0., 0., 0., 0., 0., 1.]]
    types['VSD']= [[1., 0., 0., 0., 0., 0.]]
    types['AVSD']=[[0., 1., 0., 0., 0., 0.]]
    types['ToF']= [[0., 0., 1., 0., 0., 0.]]
    types['TGA']= [[0., 0., 0., 1., 0., 0.]]
    types['TGA+VSD'] = [[0., 1., 0., 1., 0., 0.]]
    types['ToF+VSD'] = [[0., 1., 1., 0., 0., 0.]]
    types['Normal'] = [[0., 0., 0., 0., 0., 0.]]
    types['ToF+CA'] = [[0., 0., 1., 0., 1., 0.]]
    types['TGA+CA'] = [[0., 0., 0., 1., 1., 0.]]
    types['ToF+PuA'] = [[0., 0., 1., 0., 0., 1.]]
    types['VSD+CA'] = [[1., 0., 0., 0., 1., 0.]]
    types['ToF+CA+PuA'] = [[0., 0., 1., 0., 1., 1.]]
    types['ToF+VSD+CA+PuA'] = [[1., 0., 1., 0., 1., 1.]]
    types['ToF+TGA'] = [[0., 0., 1., 1., 0., 0.]]
    filename = data['filename']
    for k in types.keys():
        z_t = net.type_encoder(torch.from_numpy(np.array(types[k]).astype(np.float32)).to(device))
        sdf = tester.z2voxel(z_s, z_t, net, num_blocks=num_block, out_block=-1, get_tmplt_coords=False)
        sdf_noCorr = tester.z2voxel(z_s, z_t, net, num_blocks=num_block, out_block=0, get_tmplt_coords=False)
        io_utils.write_sdf_to_vtk_mesh(sdf, os.path.join(cfg['data']['train_output_dir'], '{}_pred_t{}.vtp'.format(filename[0], k)), thresh) 
        io_utils.write_sdf_to_vtk_mesh(sdf_noCorr, os.path.join(cfg['data']['train_output_dir'], '{}_pred_noCorr_t{}.vtp'.format(filename[0], k)), thresh) 

def interpolate_type_and_shape(net, data_curr, data_prev, cfg, lat_vecs=None, interval=5,num_block=1, thresh=0.5):
    chd_type_curr = data_curr['chd_type'].to(device)
    filename_curr = data_curr['filename'][0]
    chd_type_prev = data_prev['chd_type'].to(device)
    filename_prev = data_prev['filename'][0]
    type_curr = net.type_encoder(chd_type_curr)
    type_prev = net.type_encoder(chd_type_prev)
    if cfg['net']['encoder']:
        shape_curr = net.encoder(data_curr['image'].to(device))
        shape_prev = net.encoder(data_prev['image'].to(device))
    else:
        shape_curr = lat_vecs(data_curr['idx'].to(device)).view(1, cfg['net']['z_s_dim'], 2, 2, 2)
        shape_prev = lat_vecs(data_prev['idx'].to(device)).view(1, cfg['net']['z_s_dim'], 2, 2, 2)

    factor = np.linspace(0., 1., interval)
    # interpolate type
    for i, f in enumerate(factor):
        z_t = type_prev + (type_curr - type_prev) * f
        z_s = shape_prev 
        sdf = tester.z2voxel(z_s, z_t, net, num_blocks=num_block, out_block=-1, get_tmplt_coords=False)
        sdf_noCorr = tester.z2voxel(z_s, z_t, net, num_blocks=num_block, out_block=0, get_tmplt_coords=False)
        io_utils.write_sdf_to_vtk_mesh(sdf, os.path.join(cfg['data']['train_output_dir'], '{}_to_{}_pred_tInterp{}.vtp'.format(filename_prev, filename_curr, i)), thresh) 
        io_utils.write_sdf_to_vtk_mesh(sdf_noCorr, os.path.join(cfg['data']['train_output_dir'], '{}_to_{}_pred_noCorr_tInterp{}.vtp'.format(filename_prev, filename_curr, i)), thresh) 

    # interpolate shape
    for i, f in enumerate(factor):
        z_t = type_prev
        z_s = shape_prev + (shape_curr - shape_prev) * f
        sdf = tester.z2voxel(z_s, z_t, net, num_blocks=num_block, out_block=-1, get_tmplt_coords=False)
        sdf_noCorr = tester.z2voxel(z_s, z_t, net, num_blocks=num_block, out_block=0, get_tmplt_coords=False)
        io_utils.write_sdf_to_vtk_mesh(sdf, os.path.join(cfg['data']['train_output_dir'], '{}_to_{}_pred_sInterp{}.vtp'.format(filename_prev, filename_curr, i)), thresh) 
        io_utils.write_sdf_to_vtk_mesh(sdf_noCorr, os.path.join(cfg['data']['train_output_dir'], '{}_to_{}_pred_noCorr_sInterp{}.vtp'.format(filename_prev, filename_curr, i)), thresh) 


def get_original_prediction(net, z_s, data, num_block=1, thresh=0.5):
    chd_type = data['chd_type'].to(device)

    z_s_np = np.squeeze(z_s.detach().cpu().numpy().astype(np.float32))
    np.save(os.path.join(cfg['data']['train_output_dir'], '{}_feat.npy'.format(data['filename'][0])), z_s_np)
    
    curr_time = time.time()
    z_t = net.type_encoder(chd_type)
    total_time = time.time() - curr_time
    
    sdf, out_points_list = tester.z2voxel(z_s, z_t, net, num_blocks=num_block, out_block=-1, get_tmplt_coords=True)
    curr_time = time.time()
    
    sdf_noCorr = tester.z2voxel(z_s, z_t, net, num_blocks=num_block, out_block=0, get_tmplt_coords=False)
    total_time += time.time() - curr_time
    print("TIME: ", total_time)

    io_utils.write_sdf_to_vtk_mesh(sdf_noCorr, os.path.join(cfg['data']['train_output_dir'], '{}_pred_noCorr.vtp'.format(data['filename'][0])), thresh) 
    sdf_diff = sdf - sdf_noCorr # FINAL - LAST SDF
    io_utils.write_sdf_to_vtk_mesh(sdf, os.path.join(cfg['data']['train_output_dir'], '{}_pred.vtp'.format(data['filename'][0])), thresh) 
    io_utils.write_vtk_image(sdf_diff, os.path.join(cfg['data']['train_output_dir'], '{}_pred_ds.vti'.format(data['filename'][0])))
    type_sdf = tester.z2voxel(z_s, z_t, net, num_blocks=num_block, out_type=True)
    io_utils.write_sdf_to_vtk_mesh(type_sdf, os.path.join(cfg['data']['train_output_dir'], '{}_pred_type.vtp'.format(data['filename'][0])), thresh) 
    dice = metrics.dice_score_from_sdf(sdf[1:-1, 1:-1, 1:-1], data['y'].numpy()[0])
    z_v = z_t.cpu().detach().numpy()
    return z_v, dice, total_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--config')
    args = parser.parse_args()

    THRESH = 0.5
    MODE = ['aligned_train']
    num_block=1
    with open(args.config, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)

    if not os.path.exists(cfg['data']['output_dir']):
        os.makedirs(cfg['data']['output_dir'])
    
    tester = Tester(device, cell_grid_size=2, out_dim=cfg['net']['out_dim'])
    dice_score_list, time_list = [], []
    z_vector_list = {}

    # TRAINING ACCURACY
    if cfg['net']['encoder']:
        train = dataset.ImgSDFDataset(cfg['data']['train_dir'], cfg['train']['n_smpl_pts'], chd_info=cfg['data']['chd_info'], mode=MODE, use_aug=False)
    else:
        train = dataset.SDFDataset(cfg['data']['train_dir'], cfg['train']['n_smpl_pts'], chd_info=cfg['data']['chd_info'], mode=MODE, use_aug=True)
    dataloader_test = DataLoader(train, batch_size=1, shuffle=False, pin_memory=True)
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
                latent_dim=cfg['net']['latent_dim'],
                ins_norm=cfg['net']['ins_norm'])
    else:
        net = SDF4CHD(in_dim=0, \
                out_dim=cfg['net']['out_dim'], \
                num_types=len(cfg['data']['chd_info']['types']), \
                z_t_dim=cfg['net']['z_t_dim'], \
                z_s_dim=cfg['net']['z_s_dim'], \
                type_mlp_num=cfg['net']['type_mlp_num'],\
                ds_mlp_num=cfg['net']['ds_mlp_num'],\
                dx_mlp_num=cfg['net']['dx_mlp_num'], \
                latent_dim=cfg['net']['latent_dim'], \
                ins_norm=cfg['net']['ins_norm'])
        # initialize Z_s
        lat_vecs = torch.nn.Embedding(len(train), cfg['net']['z_s_dim']*2*2*2, max_norm=50.).to(device)
        lat_vecs.load_state_dict(torch.load(os.path.join(cfg['data']['output_dir'], 'code{}.pt'.format(args.epoch)), map_location=torch.device('cpu'))['latent_codes'])
    net.to(device)
    net.load_state_dict(torch.load(os.path.join(cfg['data']['output_dir'], 'net_{}.pt'.format(args.epoch)), map_location=torch.device('cpu'))['state_dict'])

    cfg['data']['train_output_dir'] = os.path.join(cfg['data']['output_dir'], MODE[0] + '_{}'.format(args.epoch))
    if not os.path.exists(cfg['data']['train_output_dir']):
        os.makedirs(cfg['data']['train_output_dir'])
    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            print(i)
            # pass for augmented copies when using auto-decoder
            if (not cfg['net']['encoder']) and (data['filename'][0][-5:] != 'image'):
                continue
            if cfg['net']['encoder']:
                z_s = net.encoder(data['image'].to(device))
            else:
                z_s = lat_vecs(data['idx'].to(device)).view(1, cfg['net']['z_s_dim'], 2, 2, 2)
            z_vector, dice, total_time = get_original_prediction(net, z_s, data, num_block=num_block, thresh=THRESH)
            z_vector_list[data['filename'][0]] = z_vector
            dice_score_list.append(dice)
            time_list.append(total_time)
        print("TIME AVG: ", np.mean(np.array(total_time)))
    pickle.dump(z_vector_list, open(os.path.join(cfg['data']['train_output_dir'], 'zs_lists'), 'wb'))
    io_utils.write_scores(os.path.join(cfg['data']['train_output_dir'], 'dice.csv'), dice_score_list)
    print(dice_score_list)
    # TESTING ACCURACY
    MODE = ['aligned_test']
    cfg['data']['test_output_dir'] = os.path.join(cfg['data']['output_dir'], MODE[0]+ '_{}'.format(args.epoch))
    if not os.path.exists(cfg['data']['test_output_dir']):
        os.makedirs(cfg['data']['test_output_dir'])
    fit_testdata(net, cfg, mode=MODE, iter_num=200)
    # to deform mesh created from type-SDF
    #fit_testdata(net, cfg, mode=['aligned_test'], iter_num=200,  output_mode='mesh')
    
    # GENERATION
    with torch.no_grad():
        data_prev = None
        for i, data in enumerate(dataloader_test):
            print(i)
            if (not cfg['net']['encoder']) and (data['filename'][0][-5:] != 'image'):
                continue
            sample_shape_space(net, data, cfg, lat_vecs=lat_vecs, num_copies=5, num_block=num_block, thresh=THRESH, get_img=False)
            sample_type_space(net, data, cfg, lat_vecs=lat_vecs, num_block=num_block, thresh=THRESH)
            if data_prev is not None:
                interpolate_type_and_shape(net, data, data_prev, cfg, lat_vecs=lat_vecs,interval=10,num_block=num_block, thresh=THRESH) 
            data_prev = data
