import os
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from img_network import Isensee3DUNetEncoder
from io_utils import *

def act(x):
    x = torch.max(torch.min(x, x * 0.05 + 0.99), x * 0.05)
    return x

def positional_encoding(
    tensor, num_encoding_functions=4, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

class TYPEDecoder(torch.nn.Module):
    def __init__(self, z_dim, df_dim, mlp_num=6, out_dim=1, point_dim=3):
        super(TYPEDecoder, self).__init__()
        self.mlp_num = mlp_num 
        self.linear_1 = nn.Linear(z_dim+point_dim, df_dim, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        for i in range(mlp_num):
            lin = nn.Linear(df_dim, df_dim, bias=True)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.constant_(lin.bias, 0)
            setattr(self, 'lin_{}'.format(i), lin)
        self.linear_out = nn.Linear(df_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.constant_(self.linear_out.bias, 0)

    def forward(self, z, points):
        point_z = torch.cat([positional_encoding(points, num_encoding_functions=6), torch.tile(z.unsqueeze(1), (1, points.shape[1], 1))], axis=-1)
        x = self.linear_1(point_z)
        x = F.leaky_relu(x, negative_slope=0.01, inplace=True)

        for i in range(self.mlp_num):
            x = getattr(self, 'lin_{}'.format(i))(x)
            x = F.leaky_relu(x, negative_slope=0.01, inplace=True) 
            #print("X2: ", torch.max(x))
        x = self.linear_out(x)
        return x

class SHAPEDecoder(nn.Module):
    def __init__(self, z_s_dim, z_t_dim, df_dim, num_layers, ins_norm=False):
        super(SHAPEDecoder, self).__init__()
        self.fc1_dx = nn.Linear(z_s_dim+z_t_dim+144, df_dim, bias=True)
        self.fc0_x = nn.Linear(3, 16, bias=True)
        self.fc1_x = nn.Linear(16, 16, bias=True)
        for i in range(num_layers):
            setattr(self, 'shape_mlp_{}'.format(i), nn.Linear(df_dim, df_dim, bias=True))
        self.fc_out_dx = nn.Linear(df_dim, 3, bias=True)
        self.num_layers = num_layers
        if ins_norm:
            self.norm = torch.nn.InstanceNorm3d(num_features=im_dim_dx)
        self.ins_norm = ins_norm

    def deform(self, points, points_type, grid):
        points_s = (points).unsqueeze(1).unsqueeze(1)   # (B, 1, 1, 4096, 3)
        points_z = F.grid_sample(grid, points_s, padding_mode='border', align_corners=True)  # (B, 32, 1, 1, 4096)
        points_z = points_z.squeeze(2).squeeze(2).permute(0, 2, 1)

        xyz_feat = self.fc1_x(F.leaky_relu(self.fc0_x(points), 0.2))
        points_f = torch.cat([positional_encoding(xyz_feat), points_type, points_z], dim=-1)
        points_f = F.leaky_relu(self.fc1_dx(points_f), 0.2)
        for i in range(self.num_layers):
            lin = getattr(self, 'shape_mlp_{}'.format(i))
            points_f = F.leaky_relu(lin(points_f), 0.2)
        dx = self.fc_out_dx(points_f)
        return dx

    def flow(self, V, x, x_type, solver='euler', step_size=0.25, T=1.0, block=0, inverse=False):
            
        h = step_size
        N = int(T/h)
        
        fac = -1. if inverse else 1.
        magnitude = 0.
        if solver == 'euler':
            # forward Euler method
            for n in range(N):
                dx = self.deform(x, x_type, V)
                x = x + fac * h * dx
                magnitude += torch.mean(dx**2) 
        if solver == 'midpoint':
            # midpoint method
            for n in range(N):
                dx1 = self.deform(x, x_type, V)
                dx2 = self.deform(x + fac*h*dx1/2, x_type, V)
                x = x + fac * h * dx2
                magnitude += torch.mean(dx2**2)
        if solver == 'heun':
            # Heun's method
            for n in range(N):
                dx1 = self.deform(x, x_type, V)
                dx2 = self.deform(x + fac*h*dx1, x_type, V)
                x = x + fac * h * (dx1 + dx2) / 2
                magnitude += (torch.mean(dx1**2) + torch.mean(dx2**2))/2.
        if solver == 'rk4':
            # fourth-order RK method
            for n in range(N):
                dx1 = self.deform(x, x_type, V)
                dx2 = self.deform(x + fac*h*dx1/2, x_type, V)
                dx3 = self.deform(x + fac*h*dx2/2, x_type, V)
                dx4 = self.deform(x + fac*h*dx3, x_type, V)
                x = x + fac * h * (dx1 + 2*dx2 + 2*dx3 + dx4) / 6
                magnitude += (torch.mean(dx1**2)+2.*torch.mean(dx3**2)+2.*torch.mean(dx3**2)+ torch.mean(dx1**2))/6.
        return x, magnitude 

    def forward(self, points, type_vec, shape_grid, inverse=False, step_size=0.2):
        # Flow from ground truth locations to topology space
        if self.ins_norm:
            shape_grid = self.norm(shape_grid)
        points_o, magnitude = self.flow(shape_grid, points, type_vec.unsqueeze(1).expand(-1, points.shape[1], type_vec.shape[-1]), step_size=step_size, inverse=inverse)
        return points_o, magnitude

class CorrectionDecoder(nn.Module):
    def __init__(self, z_s_dim, z_t_dim, df_dim, num_layers, ins_norm=False):
        super(CorrectionDecoder, self).__init__()
        self.fc1_ds = nn.Linear(z_s_dim+z_t_dim+39, df_dim, bias=True)
        for i in range(num_layers):
            setattr(self, 'ds_mlp_{}'.format(i), nn.Linear(df_dim, df_dim, bias=True))
        self.fc_out_ds = nn.Linear(df_dim, 1, bias=True)
        self.num_layers = num_layers
        if ins_norm:
            self.norm = torch.nn.InstanceNorm3d(num_features=z_dim_ds)
        self.ins_norm = ins_norm

    def forward(self, points, type_vec, grid):
        if self.ins_norm:
            grid = self.norm(grid)
        points_type = type_vec.unsqueeze(1).expand(-1, points.shape[1], type_vec.shape[-1])
        points_s = (points[:, :, :3]).unsqueeze(1).unsqueeze(1)   # (B, 1, 1, num_pts, 3)
        points_z = F.grid_sample(grid, points_s, padding_mode='border', align_corners=True)  # (B, z_s_dim, 1, 1, num_pts)
        points_z = points_z.squeeze(2).squeeze(2).permute(0, 2, 1)

        points_f = torch.cat([positional_encoding(points, num_encoding_functions=6), points_type, points_z], dim=-1)
        points_f = F.leaky_relu(self.fc1_ds(points_f), 0.2)
        for i in range(self.num_layers):
            lin = getattr(self, 'ds_mlp_{}'.format(i))
            points_f = F.leaky_relu(lin(points_f), 0.2)
        ds = self.fc_out_ds(points_f)
        return ds

class TypeEncoder(nn.Module):
    def __init__(self, in_dim, df_dim, mlp_num=4, out_dim=1):
        super(TypeEncoder, self).__init__()
        self.mlp_num = mlp_num 
        self.linear_1 = nn.Linear(in_dim, df_dim, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        for i in range(mlp_num):
            lin = nn.Linear(df_dim, df_dim, bias=True)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.constant_(lin.bias, 0)
            setattr(self, 'lin_{}'.format(i), lin)
        self.linear_out = nn.Linear(df_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.constant_(self.linear_out.bias, 0)

    def forward(self, z):
        x = self.linear_1(z)
        x = F.leaky_relu(x, negative_slope=0.02, inplace=True)
        for i in range(self.mlp_num):
            x = getattr(self, 'lin_{}'.format(i))(x)
            x = F.leaky_relu(x, negative_slope=0.01, inplace=True) 
        x = self.linear_out(x)
        return x

class DisentangledGridDecoder3D(nn.Module):
    def __init__(self, z_t_dim=64, \
            z_s_dim=128, \
            df_dim=512, \
            type_mlp_num=6, \
            ds_mlp_num=6, \
            dx_mlp_num=6, \
            out_dim=1):
        super(DisentangledGridDecoder3D, self).__init__()
        self.decoder = TYPEDecoder(z_dim=z_t_dim, df_dim=df_dim, mlp_num=type_mlp_num, out_dim=out_dim, point_dim=39)
        self.flow = SHAPEDecoder(z_s_dim, z_t_dim, df_dim, dx_mlp_num)
        self.correction = CorrectionDecoder(z_s_dim, z_t_dim, df_dim, ds_mlp_num) 

    def forward(self, z_s, z_t, points, get_tmplt_coords=False, add_correction=True, inverse=True):
        # first flow back to the topology space
        output_sdv, output_flow_mag = [], []
        if get_tmplt_coords:
            points_t_list = []
        points_t, magnitude = self.flow(points, z_t, z_s, inverse=inverse)
        
        if get_tmplt_coords:
            points_t_list.append(points_t)
        # get the sdv at the point locations
        points_t_sdv = self.decoder(z_t, points_t)
        
        output_sdv.append(points_t_sdv)
        
        if add_correction:
            ds = self.correction(points, z_t, z_s)
            points_t_sdv_ds = points_t_sdv + ds
            output_sdv.append(points_t_sdv_ds)
        
        if self.training:
            return {'recons': [act(o).permute(0, 2, 1) for o in output_sdv], 'flow_mag': torch.mean(magnitude**2)}
        else:
            if get_tmplt_coords:
                return [act(o) for o in output_sdv], points_t_list
            else:
                return [act(o) for o in output_sdv] 

class SDF4CHD(nn.Module):
    def __init__(self, in_dim=1, \
            out_dim=1, \
            num_types=1, \
            z_t_dim=64, \
            z_s_dim=128, \
            type_mlp_num=6, \
            ds_mlp_num=6, \
            dx_mlp_num=6, \
            latent_dim=512):
        super(SDF4CHD, self).__init__()
        self.decoder = DisentangledGridDecoder3D(z_t_dim=z_t_dim, z_s_dim=z_s_dim, df_dim=latent_dim, type_mlp_num=type_mlp_num, ds_mlp_num=ds_mlp_num, dx_mlp_num=dx_mlp_num, out_dim=out_dim)
        self.type_encoder = TypeEncoder(in_dim=num_types, df_dim=latent_dim, mlp_num=type_mlp_num, out_dim=z_t_dim)
        if in_dim>0:
            self.encoder = Isensee3DUNetEncoder(in_channels=in_dim, base_n_filter=16, z_dim=z_s_dim, n_conv_blocks=5)
        self.in_dim = in_dim
    
    def forward(self, z_s, points, chd_type, get_tmplt_coords=False):
        if self.in_dim >0:
            z_s = self.encoder(z_s)
        z_t = self.type_encoder(chd_type)
        if self.training:
            outputs = self.decoder(z_s, z_t, points)
            return outputs, z_t
        else:
            if get_tmplt_coords:
                points_t_sdv_list, points_t_list = self.decoder(z_s, z_t, points)
                return points_t_sdv_list, points_t_list
            else:
                points_t_sdv_list = self.decoder(z_s, z_t, points)
                return points_t_sdv_list

class Tester:
    def __init__(self, device, cell_grid_size=4, frame_grid_size=64, out_dim=1):
        self.test_size = 32  # related to testing batch_size, adjust according to gpu memory size
        self.out_dim=out_dim
        self.cell_grid_size = cell_grid_size
        self.frame_grid_size = frame_grid_size
        self.real_size = self.cell_grid_size * self.frame_grid_size  # =256, output point-value voxel grid size in testing
        self.test_point_batch_size = self.test_size * self.test_size * self.test_size  # 32 x 32 x 32, do not change
        self.sampling_threshold = 0.5
        self.device = device

        self.get_test_coord_for_training()  # initialize self.coords
        self.get_test_coord_for_testing()  # initialize self.frame_coords

    def get_test_coord_for_training(self):
        dima = self.test_size  # 32
        dim = self.frame_grid_size  # 64
        multiplier = int(dim / dima)  # 2
        multiplier2 = multiplier * multiplier
        multiplier3 = multiplier * multiplier * multiplier
        ranges = np.arange(0, dim, multiplier, np.uint8)
        self.aux_x = np.ones([dima, dima, dima], np.uint8) * np.expand_dims(ranges, axis=(1, 2))
        self.aux_y = np.ones([dima, dima, dima], np.uint8) * np.expand_dims(ranges, axis=(0, 2))
        self.aux_z = np.ones([dima, dima, dima], np.uint8) * np.expand_dims(ranges, axis=(0, 1))
        self.coords = np.zeros([multiplier ** 3, dima, dima, dima, 3], np.float32)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 0] = self.aux_x + i
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 1] = self.aux_y + j
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 2] = self.aux_z + k
        self.coords = 2.*((self.coords.astype(np.float32) + 0.5) / dim - 0.5)
        self.coords = np.reshape(self.coords, [multiplier3, self.test_point_batch_size, 3])
        self.coords = torch.from_numpy(self.coords).to(self.device)

    def get_test_coord_for_testing(self):
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size
        self.cell_x = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_y = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_z = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_coords = np.zeros([dimf, dimf, dimf, dimc, dimc, dimc, 3], np.float32)
        self.frame_coords = np.zeros([dimf, dimf, dimf, 3], np.float32)
        self.frame_x = np.zeros([dimf, dimf, dimf], np.int32)
        self.frame_y = np.zeros([dimf, dimf, dimf], np.int32)
        self.frame_z = np.zeros([dimf, dimf, dimf], np.int32)
        for i in range(dimc):
            for j in range(dimc):
                for k in range(dimc):
                    self.cell_x[i, j, k] = i
                    self.cell_y[i, j, k] = j
                    self.cell_z[i, j, k] = k
        for i in range(dimf):
            for j in range(dimf):
                for k in range(dimf):
                    self.cell_coords[i, j, k, :, :, :, 0] = self.cell_x + i * dimc
                    self.cell_coords[i, j, k, :, :, :, 1] = self.cell_y + j * dimc
                    self.cell_coords[i, j, k, :, :, :, 2] = self.cell_z + k * dimc
                    self.frame_coords[i, j, k, 0] = i
                    self.frame_coords[i, j, k, 1] = j
                    self.frame_coords[i, j, k, 2] = k
                    self.frame_x[i, j, k] = i
                    self.frame_y[i, j, k] = j
                    self.frame_z[i, j, k] = k

        self.cell_coords = 2.*((self.cell_coords.astype(np.float32) + 0.5) / self.real_size - 0.5)
        self.cell_coords = np.reshape(self.cell_coords, [dimf, dimf, dimf, dimc * dimc * dimc, 3])
        self.cell_x = np.reshape(self.cell_x, [dimc * dimc * dimc])
        self.cell_y = np.reshape(self.cell_y, [dimc * dimc * dimc])
        self.cell_z = np.reshape(self.cell_z, [dimc * dimc * dimc])
        self.frame_x = np.reshape(self.frame_x, [dimf * dimf * dimf])
        self.frame_y = np.reshape(self.frame_y, [dimf * dimf * dimf])
        self.frame_z = np.reshape(self.frame_z, [dimf * dimf * dimf])
        self.frame_coords = 2.*((self.frame_coords.astype(np.float32) + 0.5) / dimf - 0.5)
        self.frame_coords = np.reshape(self.frame_coords, [dimf * dimf * dimf, 3])
        self.frame_coords = torch.from_numpy(self.frame_coords).to(self.device)
    
    def z2voxel(self, z_s, z_t, network, num_blocks, out_block=-1, out_type=False, get_tmplt_coords=False, add_correction=True):
        # NEED TO CHANGE FOR TYPE ENCODER
        network.eval()
        model_float = np.zeros([self.real_size + 2, self.real_size + 2, self.real_size + 2, self.out_dim], np.float32)
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size

        frame_flag = np.zeros([dimf + 2, dimf + 2, dimf + 2, self.out_dim], np.uint8)
        queue = []
        frame_batch_num = int(dimf ** 3 / self.test_point_batch_size)
        assert frame_batch_num > 0

        out_point_coords_list = [torch.zeros((0, 3)).to(z_t.device) for i in range(num_blocks)]
        # get frame grid values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            point_coord = point_coord.unsqueeze(dim=0)
            with torch.no_grad():
                if out_type:
                    model_out_ = act(network.decoder.decoder(z_t, point_coord))
                else:
                    if get_tmplt_coords:
                        model_out_, points_t_list = network.decoder(z_s, z_t, point_coord, get_tmplt_coords, add_correction=add_correction)
                        for q, (points_t, sdf) in enumerate(zip(points_t_list, model_out_)):
                            out_point_coords_list[q] = torch.cat([out_point_coords_list[q], points_t[torch.any(sdf>self.sampling_threshold, dim=-1)]], dim=0)
                    else:
                        model_out_ = network.decoder(z_s, z_t, point_coord, add_correction=add_correction)
                    model_out_ = model_out_[out_block]
                model_out = model_out_.detach().cpu().numpy()[0]
            x_coords = self.frame_x[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            y_coords = self.frame_y[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            z_coords = self.frame_z[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            frame_flag[x_coords + 1, y_coords + 1, z_coords + 1, :] = np.reshape((model_out > self.sampling_threshold).astype(np.uint8),
                                                                              [self.test_point_batch_size, self.out_dim])

        # get queue and fill up ones
        for i in range(1, dimf + 1):
            for j in range(1, dimf + 1):
                for k in range(1, dimf + 1):
                    maxv = np.max(frame_flag[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                    minv = np.min(frame_flag[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                    if maxv != minv:
                        queue.append((i, j, k))
                    elif maxv == 1:
                        x_coords = self.cell_x + (i - 1) * dimc
                        y_coords = self.cell_y + (j - 1) * dimc
                        z_coords = self.cell_z + (k - 1) * dimc
                        model_float[x_coords + 1, y_coords + 1, z_coords + 1, :] = 1.0

        # print("running queue:", len(queue))
        cell_batch_size = dimc ** 3
        cell_batch_num = int(self.test_point_batch_size / cell_batch_size)
        assert cell_batch_num > 0
        # run queue
        while len(queue) > 0:
            batch_num = min(len(queue), cell_batch_num)
            point_list = []
            cell_coords = []
            for i in range(batch_num):
                point = queue.pop(0)
                point_list.append(point)
                cell_coords.append(self.cell_coords[point[0] - 1, point[1] - 1, point[2] - 1])
            cell_coords = np.concatenate(cell_coords, axis=0)
            cell_coords = np.expand_dims(cell_coords, axis=0)
            cell_coords = torch.from_numpy(cell_coords)
            cell_coords = cell_coords.to(self.device)
            with torch.no_grad():
                if out_type:
                    model_out_batch_ = act(network.decoder.decoder(z_t, cell_coords))
                else:
                    if get_tmplt_coords:
                        model_out_batch_, points_t_list = network.decoder(z_s, z_t, cell_coords, get_tmplt_coords, add_correction=add_correction)
                        for q, (points_t, sdf) in enumerate(zip(points_t_list, model_out_batch_)):
                            out_point_coords_list[q] = torch.cat([out_point_coords_list[q], points_t[torch.any(sdf>self.sampling_threshold, dim=-1)]], dim=0)
                    else:
                        model_out_batch_ = network.decoder(z_s, z_t, cell_coords, add_correction=add_correction)
                    model_out_batch_ = model_out_batch_[out_block]
                model_out_batch = model_out_batch_.detach().cpu().numpy()[0]
            for i in range(batch_num):
                point = point_list[i]
                model_out = model_out_batch[i * cell_batch_size:(i + 1) * cell_batch_size, :]
                x_coords = self.cell_x + (point[0] - 1) * dimc
                y_coords = self.cell_y + (point[1] - 1) * dimc
                z_coords = self.cell_z + (point[2] - 1) * dimc
                model_float[x_coords + 1, y_coords + 1, z_coords + 1, :] = model_out

                if np.max(model_out) > self.sampling_threshold:
                    for i in range(-1, 2):
                        pi = point[0] + i
                        if pi <= 0 or pi > dimf:
                            continue
                        for j in range(-1, 2):
                            pj = point[1] + j
                            if pj <= 0 or pj > dimf:
                                continue
                            for k in range(-1, 2):
                                pk = point[2] + k
                                if pk <= 0 or pk > dimf:
                                    continue
                                if frame_flag[pi, pj, pk].all() == 0:
                                    frame_flag[pi, pj, pk, :] = 1
                                    queue.append((pi, pj, pk))
        if get_tmplt_coords:
            return model_float, out_point_coords_list
        else:
            return model_float
    def deform_image(self, z_s_original, z_s_sampled, z_t, img, network, num_blocks, out_block=-1, order=1):
        # NEED TO CHANGE FOR TYPE ENCODER
        network.eval()
        model_float = np.zeros([self.real_size + 2, self.real_size + 2, self.real_size + 2, img.shape[1]], np.float32)
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size

        frame_flag = np.zeros([dimf + 2, dimf + 2, dimf + 2, img.shape[1]], np.uint8)
        queue = []
        frame_batch_num = int(dimf ** 3 / self.test_point_batch_size)
        assert frame_batch_num > 0

        out_point_coords_list = [torch.zeros((0, 3)).to(z_t.device) for i in range(num_blocks)]
        # get frame grid values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            point_coord = point_coord.unsqueeze(dim=0)
            with torch.no_grad():
                _, points_t_list = network.decoder(z_s_sampled, z_t, point_coord, get_tmplt_coords=True, add_correction=False, inverse=True)
                _, points_new_s_list = network.decoder(z_s_original, z_t, points_t_list[0], get_tmplt_coords=True, add_correction=False, inverse=False)
                points_new_s = points_new_s_list[0].unsqueeze(1).unsqueeze(1)
                model_out_ = F.grid_sample(img, points_new_s, padding_mode='border', align_corners=True, mode='bilinear' if order==1 else 'nearest')
                model_out_ = model_out_.squeeze(2).squeeze(2).permute(0, 2, 1)
                model_out = model_out_.detach().cpu().numpy()[0]
            x_coords = self.frame_x[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            y_coords = self.frame_y[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            z_coords = self.frame_z[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            frame_flag[x_coords + 1, y_coords + 1, z_coords + 1, :] = np.reshape((model_out > -1000.).astype(np.uint8),
                                                                              [self.test_point_batch_size, img.shape[1]])

        # get queue and fill up ones
        for i in range(1, dimf + 1):
            for j in range(1, dimf + 1):
                for k in range(1, dimf + 1):
                    maxv = np.max(frame_flag[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                    minv = np.min(frame_flag[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                    queue.append((i, j, k))
        cell_batch_size = dimc ** 3
        cell_batch_num = int(self.test_point_batch_size / cell_batch_size)
        assert cell_batch_num > 0
        # run queue
        while len(queue) > 0:
            batch_num = min(len(queue), cell_batch_num)
            point_list = []
            cell_coords = []
            for i in range(batch_num):
                point = queue.pop(0)
                point_list.append(point)
                cell_coords.append(self.cell_coords[point[0] - 1, point[1] - 1, point[2] - 1])
            cell_coords = np.concatenate(cell_coords, axis=0)
            cell_coords = np.expand_dims(cell_coords, axis=0)
            cell_coords = torch.from_numpy(cell_coords)
            cell_coords = cell_coords.to(self.device)
            with torch.no_grad():
                _, points_t_list = network.decoder(z_s_sampled, z_t, cell_coords, get_tmplt_coords=True, add_correction=False, inverse=True)
                _, points_new_s_list = network.decoder(z_s_original, z_t, points_t_list[0], get_tmplt_coords=True, add_correction=False, inverse=False)
                points_new_s = points_new_s_list[0].unsqueeze(1).unsqueeze(1)
                model_out_batch_ = F.grid_sample(img, points_new_s, padding_mode='border', align_corners=True, mode='bilinear' if order==1 else 'nearest')
                model_out_batch_ = model_out_batch_.squeeze(2).squeeze(2).permute(0, 2, 1)
                model_out_batch = model_out_batch_.detach().cpu().numpy()[0]
            for i in range(batch_num):
                point = point_list[i]
                model_out = model_out_batch[i * cell_batch_size:(i + 1) * cell_batch_size, :]
                x_coords = self.cell_x + (point[0] - 1) * dimc
                y_coords = self.cell_y + (point[1] - 1) * dimc
                z_coords = self.cell_z + (point[2] - 1) * dimc
                model_float[x_coords + 1, y_coords + 1, z_coords + 1, :] = model_out

                if np.max(model_out) > -1000.:
                    for i in range(-1, 2):
                        pi = point[0] + i
                        if pi <= 0 or pi > dimf:
                            continue
                        for j in range(-1, 2):
                            pj = point[1] + j
                            if pj <= 0 or pj > dimf:
                                continue
                            for k in range(-1, 2):
                                pk = point[2] + k
                                if pk <= 0 or pk > dimf:
                                    continue
                                if frame_flag[pi, pj, pk].all() == 0:
                                    frame_flag[pi, pj, pk, :] = 1
                                    queue.append((pi, pj, pk))
        return model_float
