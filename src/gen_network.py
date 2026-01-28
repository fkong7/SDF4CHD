import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from io_utils import *
from net_utils import *


class TYPEDecoder(torch.nn.Module):
    def __init__(
        self,
        num_types,
        z_dim,
        df_dim,
        mlp_num=6,
        out_dim=1,
        point_dim=3,
        bias=True,
        lip_reg=True,
        use_diag=False,
    ):
        super(TYPEDecoder, self).__init__()
        self.mlp_num = mlp_num
        if lip_reg:
            if not use_diag:
                self.linear_1 = LipLinearLayer(z_dim + point_dim, df_dim, bias=True)
            else:
                self.linear_1 = LipLinearLayer(point_dim + num_types, df_dim, bias=True)
        else:
            self.linear_1 = nn.Linear(z_dim, df_dim, bias=True)
        for i in range(mlp_num):
            if lip_reg:
                lin = LipLinearLayer(df_dim, df_dim, bias=True)
            else:
                lin = nn.Linear(df_dim, df_dim, bias=True)
            setattr(self, "lin_{}".format(i), lin)
        if lip_reg:
            self.linear_out = LipLinearLayer(df_dim, out_dim, bias=True)
        else:
            self.linear_out = nn.Linear(df_dim, out_dim, bias=True)
        self.use_diag = use_diag

    def forward(self, z, points):
        if not self.use_diag:
            points_tz = torch.tile(z.unsqueeze(1), (1, points.shape[1], 1))
            point_z = torch.cat(
                [positional_encoding(points, num_encoding_functions=6), points_tz],
                dim=-1,
            )
        else:
            points_tz = torch.tile(z.unsqueeze(1), (1, points.shape[1], 1))
            point_z = torch.cat(
                [positional_encoding(points, num_encoding_functions=6), points_tz],
                dim=-1,
            )

        x = self.linear_1(point_z)
        x = F.leaky_relu(x, negative_slope=0.01, inplace=True)

        for i in range(self.mlp_num):
            x = getattr(self, "lin_{}".format(i))(x)
            x = F.leaky_relu(x, negative_slope=0.01, inplace=True)
        x = self.linear_out(x)
        return x


class SHAPEDecoder(nn.Module):
    def __init__(
        self,
        z_s_dim,
        z_t_dim,
        df_dim,
        num_layers,
        type_decoder,
        ins_norm=False,
        out_dim=1,
        step_size=0.2,
        div_loss=False,
    ):
        super(SHAPEDecoder, self).__init__()
        self.fc1_dx = nn.Linear(z_s_dim, df_dim, bias=True)
        for i in range(num_layers):
            setattr(
                self, "shape_mlp_{}".format(i), nn.Linear(df_dim, df_dim, bias=True)
            )
        self.fc_out_dx = nn.Linear(df_dim, 3, bias=True)
        self.num_layers = num_layers
        if ins_norm:
            self.norm = torch.nn.InstanceNorm3d(num_features=z_s_dim)
        self.ins_norm = ins_norm
        self.type_decoder = type_decoder
        self.step_size = step_size
        dx = torch.tensor([1.0 / 128.0, 0.0, 0.0], dtype=torch.float32).view(1, 1, 3)
        dy = torch.tensor([0.0, 1.0 / 128.0, 0.0], dtype=torch.float32).view(1, 1, 3)
        dz = torch.tensor([0.0, 0.0, 1.0 / 128.0], dtype=torch.float32).view(1, 1, 3)
        self.register_buffer("dx", dx)
        self.register_buffer("dy", dy)
        self.register_buffer("dz", dz)
        self.div_loss = div_loss

    def deform(self, points, points_type, grid):
        points_s = (points).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 4096, 3)
        points_z = F.grid_sample(
            grid, points_s, padding_mode="border", align_corners=True
        )  # (B, 32, 1, 1, 4096)
        points_z = points_z.squeeze(2).squeeze(2).permute(0, 2, 1)

        points_f = positional_encoding(points, num_encoding_functions=4) * points_z
        points_f = F.leaky_relu(self.fc1_dx(points_f), 0.2)
        for i in range(self.num_layers):
            lin = getattr(self, "shape_mlp_{}".format(i))
            points_f = F.leaky_relu(lin(points_f), 0.2)
        dx = self.fc_out_dx(points_f)
        return dx

    def flow(self, V, x, x_type, solver="euler", T=1.0, block=0, inverse=False):
        h = self.step_size
        N = int(T / h)

        fac = -1.0 if inverse else 1.0
        div_integral, grad_mag = 0.0, 0.0
        if solver == "euler":
            # forward Euler method
            for n in range(N):
                dx = self.deform(x, x_type, V)
                x = x + fac * h * dx
                if self.training:
                    grad_mag += torch.mean(dx**2, dim=-1)
                    if self.div_loss:
                        # get divergence integral
                        u_x = (
                            self.deform(x + self.dx, x_type, V)[:, :, 0]
                            - self.deform(x - self.dx, x_type, V)[:, :, 0]
                        ) / (2.0 * self.dx[:, :, 0])
                        u_y = (
                            self.deform(x + self.dy, x_type, V)[:, :, 1]
                            - self.deform(x - self.dy, x_type, V)[:, :, 1]
                        ) / (2.0 * self.dy[:, :, 1])
                        u_z = (
                            self.deform(x + self.dz, x_type, V)[:, :, 2]
                            - self.deform(x - self.dz, x_type, V)[:, :, 2]
                        ) / (2.0 * self.dz[:, :, 2])
                        div = u_x + u_y + u_z
                        div = torch.clamp(div, min=-3.0)
                        div_integral += (-div).exp().mean() * h

        return x, div_integral, grad_mag

    def forward(self, points, type_vec, shape_grid, inverse=False):
        # Flow from ground truth locations to topology space
        if self.ins_norm:
            shape_grid = self.norm(shape_grid)
        points_o, div_integral, grad_mag = self.flow(
            shape_grid, points, type_vec, inverse=inverse
        )
        return points_o, div_integral, grad_mag


class CorrectionDecoder(nn.Module):
    def __init__(self, z_s_dim, z_t_dim, df_dim, num_layers, ins_norm=False, out_dim=1):
        super(CorrectionDecoder, self).__init__()
        self.fc1_ds = nn.Linear(z_s_dim, df_dim, bias=True)
        for i in range(num_layers):
            setattr(self, "ds_mlp_{}".format(i), nn.Linear(df_dim, df_dim, bias=True))
        self.fc_out_ds = nn.Linear(df_dim, 1, bias=True)
        self.num_layers = num_layers
        if ins_norm:
            self.norm = torch.nn.InstanceNorm3d(num_features=z_s_dim)
        self.ins_norm = ins_norm

    def forward(self, points, points_sdf, grid):
        if self.ins_norm:
            grid = self.norm(grid)
        points_s = (points[:, :, :3]).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, num_pts, 3)
        points_z = F.grid_sample(
            grid, points_s, padding_mode="border", align_corners=True
        )  # (B, z_s_dim, 1, 1, num_pts)
        points_z = points_z.squeeze(2).squeeze(2).permute(0, 2, 1)

        points_f = positional_encoding(points, num_encoding_functions=4) * points_z

        points_f = F.leaky_relu(self.fc1_ds(points_f), 0.2)
        for i in range(self.num_layers):
            lin = getattr(self, "ds_mlp_{}".format(i))
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
            setattr(self, "lin_{}".format(i), lin)
        self.linear_out = nn.Linear(df_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.constant_(self.linear_out.bias, 0)
        self.out_dim = out_dim

    def forward(self, z):
        x = self.linear_1(z)
        x = F.leaky_relu(x, negative_slope=0.02, inplace=True)
        for i in range(self.mlp_num):
            x = getattr(self, "lin_{}".format(i))(x)
            x = F.leaky_relu(x, negative_slope=0.01, inplace=True)
        x = self.linear_out(x)
        return x


class DisentangledGridDecoder3D(nn.Module):
    def __init__(
        self,
        num_types=1,
        z_t_dim=64,
        z_s_dim=128,
        df_dim=512,
        type_mlp_num=6,
        ds_mlp_num=6,
        dx_mlp_num=6,
        out_dim=1,
        ins_norm=False,
        type_bias=True,
        lip_reg=True,
        step_size=0.2,
        use_diag=False,
        div_loss=False,
        act_func=act,
    ):
        super(DisentangledGridDecoder3D, self).__init__()
        self.decoder = TYPEDecoder(
            num_types=num_types,
            z_dim=z_t_dim,
            df_dim=df_dim,
            mlp_num=type_mlp_num,
            out_dim=out_dim,
            point_dim=39,
            bias=type_bias,
            lip_reg=lip_reg,
            use_diag=use_diag,
        )
        self.flow = SHAPEDecoder(
            z_s_dim,
            z_t_dim,
            df_dim,
            dx_mlp_num,
            type_decoder=self.decoder,
            ins_norm=ins_norm,
            out_dim=out_dim,
            step_size=step_size,
            div_loss=div_loss,
        )
        self.correction = CorrectionDecoder(
            z_s_dim, z_t_dim, df_dim, ds_mlp_num, ins_norm=ins_norm, out_dim=out_dim
        )
        self.act = act_func

    def forward(
        self,
        z_s,
        z_s_ds,
        z_t,
        points,
        get_tmplt_coords=False,
        add_correction=True,
        inverse=True,
    ):
        # first flow back to the topology space
        output_sdv, output_flow_mag = [], []
        if get_tmplt_coords:
            points_t_list = []
        points_t, div_integral, grad_mag = self.flow(points, z_t, z_s, inverse=inverse)
        points_t = F.tanh(points_t)
        if get_tmplt_coords:
            points_t_list.append(points_t)
        # get the sdv at the point locations
        points_t_sdv = self.decoder(z_t, points_t)

        output_sdv.append(points_t_sdv)

        if add_correction:
            ds = self.correction(points, self.act(points_t_sdv), z_s_ds)
            points_t_sdv_ds = points_t_sdv + ds
            output_sdv.append(points_t_sdv_ds)

        if self.training:
            return {
                "recons": [self.act(o).permute(0, 2, 1) for o in output_sdv],
                "div_integral": div_integral,
                "grad_mag": grad_mag,
            }
        else:
            if get_tmplt_coords:
                return [self.act(o) for o in output_sdv], points_t_list
            else:
                return [self.act(o) for o in output_sdv]


class SDF4CHD(nn.Module):
    def __init__(
        self,
        in_dim=1,
        out_dim=1,
        num_types=1,
        z_t_dim=64,
        z_s_dim=128,
        type_mlp_num=6,
        ds_mlp_num=6,
        dx_mlp_num=6,
        latent_dim=512,
        ins_norm=False,
        type_bias=True,
        lip_reg=True,
        step_size=0.2,
        use_diag=False,
        div_loss=False,
        act_func=act,
    ):
        super(SDF4CHD, self).__init__()
        if not use_diag:
            self.type_encoder = TypeEncoder(
                in_dim=num_types,
                df_dim=latent_dim,
                mlp_num=type_mlp_num,
                out_dim=z_t_dim,
            )
        self.decoder = DisentangledGridDecoder3D(
            num_types=num_types,
            z_t_dim=z_t_dim,
            z_s_dim=z_s_dim,
            df_dim=latent_dim,
            type_mlp_num=type_mlp_num,
            ds_mlp_num=ds_mlp_num,
            dx_mlp_num=dx_mlp_num,
            out_dim=out_dim,
            ins_norm=ins_norm,
            lip_reg=lip_reg,
            step_size=step_size,
            use_diag=use_diag,
            div_loss=div_loss,
            act_func=act_func,
        )
        self.in_dim = in_dim
        self.use_diag = use_diag
        self.act = act_func

    def set_div_loss(self, mode: bool = True):
        self.div_loss = mode
        for m in self.modules():
            if hasattr(m, "div_loss"):
                m.div_loss = mode

    def forward(self, z_s, z_s_ds, points, chd_type, get_tmplt_coords=False):
        if not self.use_diag:
            z_t = self.type_encoder(chd_type)
        else:
            z_t = chd_type
        if self.training:
            outputs = self.decoder(z_s, z_s_ds, z_t, points)
            return outputs, z_t
        else:
            if get_tmplt_coords:
                points_t_sdv_list, points_t_list = self.decoder(
                    z_s, z_s_ds, z_t, points
                )
                return points_t_sdv_list, points_t_list
            else:
                points_t_sdv_list = self.decoder(z_s, z_s_ds, z_t, points)
                return points_t_sdv_list


class SDF4CHDTester(Tester):

    def z2voxel(
        self,
        z_s,
        z_s_ds,
        z_t,
        network,
        num_blocks,
        out_block=-1,
        out_type=False,
        get_tmplt_coords=False,
        add_correction=True,
    ):
        network.eval()
        if self.binary:
            model_float = np.zeros(
                [
                    self.real_size + 2,
                    self.real_size + 2,
                    self.real_size + 2,
                    self.out_dim,
                ],
                np.float32,
            )
        else:
            model_float = np.ones(
                [
                    self.real_size + 2,
                    self.real_size + 2,
                    self.real_size + 2,
                    self.out_dim,
                ],
                np.float32,
            )
        dimc = self.cell_grid_size  # 4^3 cube
        dimf = self.frame_grid_size  # 64^3 cube

        frame_flag = np.zeros([dimf + 2, dimf + 2, dimf + 2, self.out_dim], np.uint8)
        queue = []
        frame_batch_num = int(dimf**3 / self.test_point_batch_size)
        assert frame_batch_num > 0

        out_point_coords_list = [
            torch.zeros((0, 3)).to(z_t.device) for i in range(num_blocks)
        ]
        # get frame grid values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[
                i * self.test_point_batch_size : (i + 1) * self.test_point_batch_size
            ]
            point_coord = point_coord.unsqueeze(dim=0)
            with torch.no_grad():
                if out_type:
                    model_out_ = network.decoder.decoder(z_t, point_coord.flip(-1))
                    if not self.binary:
                        model_out_ = model_out_ * -1.0 + 0.5
                    model_out_ = network.act(model_out_)
                else:
                    if get_tmplt_coords:
                        model_out_, points_t_list = network.decoder(
                            z_s,
                            z_s_ds,
                            z_t,
                            point_coord.flip(-1),
                            get_tmplt_coords,
                            add_correction=add_correction,
                        )
                        if not self.binary:
                            model_out_ = model_out_ * -1.0 + 0.5
                        for q, (points_t, sdf) in enumerate(
                            zip(points_t_list, model_out_)
                        ):
                            out_point_coords_list[q] = torch.cat(
                                [
                                    out_point_coords_list[q],
                                    points_t[
                                        torch.any(sdf > self.sampling_threshold, dim=-1)
                                    ],
                                ],
                                dim=0,
                            )
                    else:
                        model_out_ = network.decoder(
                            z_s,
                            z_s_ds,
                            z_t,
                            point_coord.flip(-1),
                            add_correction=add_correction,
                        )
                        if not self.binary:
                            model_out_ = model_out_ * -1.0 + 0.5
                    model_out_ = model_out_[out_block]
                model_out = model_out_.detach().cpu().numpy()[0]
            x_coords = self.frame_x[
                i * self.test_point_batch_size : (i + 1) * self.test_point_batch_size
            ]
            y_coords = self.frame_y[
                i * self.test_point_batch_size : (i + 1) * self.test_point_batch_size
            ]
            z_coords = self.frame_z[
                i * self.test_point_batch_size : (i + 1) * self.test_point_batch_size
            ]
            frame_flag[x_coords + 1, y_coords + 1, z_coords + 1, :] = np.reshape(
                (model_out > self.sampling_threshold).astype(np.uint8),
                [self.test_point_batch_size, self.out_dim],
            )

        # get queue and fill up ones
        for i in range(1, dimf + 1):
            for j in range(1, dimf + 1):
                for k in range(1, dimf + 1):
                    maxv = np.max(
                        frame_flag[i - 1 : i + 2, j - 1 : j + 2, k - 1 : k + 2]
                    )
                    minv = np.min(
                        frame_flag[i - 1 : i + 2, j - 1 : j + 2, k - 1 : k + 2]
                    )
                    if maxv != minv:
                        queue.append((i, j, k))
                    elif maxv == 1:
                        x_coords = self.cell_x + (i - 1) * dimc
                        y_coords = self.cell_y + (j - 1) * dimc
                        z_coords = self.cell_z + (k - 1) * dimc
                        model_float[x_coords + 1, y_coords + 1, z_coords + 1, :] = 1.0

        cell_batch_size = dimc**3
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
                cell_coords.append(
                    self.cell_coords[point[0] - 1, point[1] - 1, point[2] - 1]
                )
            cell_coords = np.concatenate(cell_coords, axis=0)
            cell_coords = np.expand_dims(cell_coords, axis=0)
            cell_coords = torch.from_numpy(cell_coords)
            cell_coords = cell_coords.to(self.device)
            with torch.no_grad():
                if out_type:
                    model_out_batch_ = network.decoder.decoder(
                        z_t, cell_coords.flip(-1)
                    )
                    if not self.binary:
                        model_out_batch_ = model_out_batch_ * -1.0 + 0.5
                    model_out_batch_ = network.act(model_out_batch_)
                else:
                    if get_tmplt_coords:
                        model_out_batch_, points_t_list = network.decoder(
                            z_s,
                            z_s_ds,
                            z_t,
                            cell_coords.flip(-1),
                            get_tmplt_coords,
                            add_correction=add_correction,
                        )
                        if not self.binary:
                            model_out_batch_ = model_out_batch_ * -1.0 + 0.5
                        for q, (points_t, sdf) in enumerate(
                            zip(points_t_list, model_out_batch_)
                        ):
                            out_point_coords_list[q] = torch.cat(
                                [
                                    out_point_coords_list[q],
                                    points_t[
                                        torch.any(sdf > self.sampling_threshold, dim=-1)
                                    ],
                                ],
                                dim=0,
                            )
                    else:
                        model_out_batch_ = network.decoder(
                            z_s,
                            z_s_ds,
                            z_t,
                            cell_coords.flip(-1),
                            add_correction=add_correction,
                        )
                        if not self.binary:
                            model_out_batch_ = model_out_batch_ * -1.0 + 0.5
                    model_out_batch_ = model_out_batch_[out_block]
                model_out_batch = model_out_batch_.detach().cpu().numpy()[0]
            for i in range(batch_num):
                point = point_list[i]
                model_out = model_out_batch[
                    i * cell_batch_size : (i + 1) * cell_batch_size, :
                ]
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

    def deform_image(
        self,
        z_s_original,
        z_s_sampled,
        z_t,
        img,
        network,
        num_blocks,
        out_block=-1,
        order=1,
    ):
        network.eval()
        model_float = np.zeros(
            [self.real_size + 2, self.real_size + 2, self.real_size + 2, img.shape[1]],
            np.float32,
        )
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size

        frame_flag = np.zeros([dimf + 2, dimf + 2, dimf + 2, img.shape[1]], np.uint8)
        queue = []
        frame_batch_num = int(dimf**3 / self.test_point_batch_size)
        assert frame_batch_num > 0

        # get frame grid values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[
                i * self.test_point_batch_size : (i + 1) * self.test_point_batch_size
            ]
            point_coord = point_coord.unsqueeze(dim=0)
            with torch.no_grad():
                points_t, _, _ = network.decoder.flow(
                    point_coord, None, z_s_sampled, inverse=True
                )
                points_new_s, _, _ = network.decoder.flow(
                    points_t, None, z_s_original, inverse=False
                )
                points_new_s = points_new_s.unsqueeze(1).unsqueeze(1)
                model_out_ = F.grid_sample(
                    img,
                    points_new_s,
                    padding_mode="border",
                    align_corners=True,
                    mode="bilinear" if order == 1 else "nearest",
                )
                model_out_ = model_out_.squeeze(2).squeeze(2).permute(0, 2, 1)
                model_out = model_out_.detach().cpu().numpy()[0]
            x_coords = self.frame_x[
                i * self.test_point_batch_size : (i + 1) * self.test_point_batch_size
            ]
            y_coords = self.frame_y[
                i * self.test_point_batch_size : (i + 1) * self.test_point_batch_size
            ]
            z_coords = self.frame_z[
                i * self.test_point_batch_size : (i + 1) * self.test_point_batch_size
            ]
            frame_flag[x_coords + 1, y_coords + 1, z_coords + 1, :] = np.reshape(
                (model_out > -1000.0).astype(np.uint8),
                [self.test_point_batch_size, img.shape[1]],
            )

        # get queue and fill up ones
        for i in range(1, dimf + 1):
            for j in range(1, dimf + 1):
                for k in range(1, dimf + 1):
                    maxv = np.max(
                        frame_flag[i - 1 : i + 2, j - 1 : j + 2, k - 1 : k + 2]
                    )
                    minv = np.min(
                        frame_flag[i - 1 : i + 2, j - 1 : j + 2, k - 1 : k + 2]
                    )
                    queue.append((i, j, k))
        cell_batch_size = dimc**3
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
                cell_coords.append(
                    self.cell_coords[point[0] - 1, point[1] - 1, point[2] - 1]
                )
            cell_coords = np.concatenate(cell_coords, axis=0)
            cell_coords = np.expand_dims(cell_coords, axis=0)
            cell_coords = torch.from_numpy(cell_coords)
            cell_coords = cell_coords.to(self.device)
            with torch.no_grad():
                points_t, _, _ = network.decoder.flow(
                    cell_coords, None, z_s_sampled, inverse=True
                )
                points_new_s, _, _ = network.decoder.flow(
                    points_t, None, z_s_original, inverse=False
                )
                points_new_s = points_new_s.unsqueeze(1).unsqueeze(1)
                model_out_batch_ = F.grid_sample(
                    img,
                    points_new_s,
                    padding_mode="border",
                    align_corners=True,
                    mode="bilinear" if order == 1 else "nearest",
                )
                model_out_batch_ = (
                    model_out_batch_.squeeze(2).squeeze(2).permute(0, 2, 1)
                )
                model_out_batch = model_out_batch_.detach().cpu().numpy()[0]
            for i in range(batch_num):
                point = point_list[i]
                model_out = model_out_batch[
                    i * cell_batch_size : (i + 1) * cell_batch_size, :
                ]
                x_coords = self.cell_x + (point[0] - 1) * dimc
                y_coords = self.cell_y + (point[1] - 1) * dimc
                z_coords = self.cell_z + (point[2] - 1) * dimc
                model_float[x_coords + 1, y_coords + 1, z_coords + 1, :] = model_out

                if np.max(model_out) > -10000000.0:
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
