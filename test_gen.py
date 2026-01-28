import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))
import argparse
import glob
import itertools
import pickle
import random
import re
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
import vtk
import yaml
from torch.utils.data import DataLoader

import dataset
import io_utils
import metrics
import net_utils
from gen_network import SDF4CHD, SDF4CHDTester
from net_utils import act
from vtk_utils.vtk_utils import *

device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")


def fit_new_type(fn, net, cfg, excel_info, type_name):
    input_data = pickle.load(open(fn, "rb"))
    filename = os.path.basename(fn).split(".")[0]
    STD = 0.01
    PT_NUM = 32768
    # register with vsd_tga type first
    type_dict = get_type_dict(excel_info, get_new_type=False)
    chd_type = np.array(type_dict["VSD_TGA"]).astype(np.float32)

    z_s, z_s_ds, z_t, _ = fit(
        fn, chd_type, net, iter_num=10, two_shape_codes=False, opt_both=False
    )
    # get data
    sdf_noCorr = tester.z2voxel(
        z_s, z_s_ds, z_t, net, num_blocks=num_block, out_block=0, get_tmplt_coords=False
    )
    io_utils.write_sdf_to_vtk_mesh(
        sdf_noCorr,
        os.path.join(
            cfg["data"]["train_output_dir"], "{}_test_pred_noCorr.vtp".format(filename)
        ),
        THRESH,
    )
    net.train()
    for param in net.parameters():
        param.requires_grad = False
    z_s.requires_grad = False
    z_s_ds.requires_grad = False
    z_t.requires_grad = True

    optimizer_t = torch.optim.Adam([z_t], lr=0.01, betas=(0.5, 0.999), weight_decay=0.0)
    scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_t, patience=15, factor=0.6, verbose=True, min_lr=1e-6
    )

    # optimize
    for k in range(300):
        optimizer_t.zero_grad()
        _, points, point_values, _ = dataset.sample_points_from_sdf(
            input_data, PT_NUM, 20, interior=False
        )
        print(points.shape, point_values.shape)
        points = points.unsqueeze(0).to(device)
        point_values = point_values.unsqueeze(0).to(device)

        outputs = net.decoder(z_s, z_s_ds, z_t, points)
        total_loss = torch.mean(
            ((outputs["recons"][1].squeeze(-1) - point_values) ** 2)
        )
        total_loss.backward()
        print(k, total_loss.item(), z_t)
        optimizer_t.step()
        scheduler_t.step(total_loss.item())
    z_t.requires_grad = False
    type_sdf = tester.z2voxel(None, None, z_t, net, num_blocks=num_block, out_type=True)
    io_utils.write_sdf_to_vtk_mesh(
        type_sdf,
        os.path.join(
            cfg["data"]["train_output_dir"], "{}_new_type.vtp".format(type_name)
        ),
        THRESH,
        keep_largest=True,
    )
    sdf_noCorr = tester.z2voxel(
        z_s, z_s_ds, z_t, net, num_blocks=num_block, out_block=0, get_tmplt_coords=False
    )
    io_utils.write_sdf_to_vtk_mesh(
        sdf_noCorr,
        os.path.join(
            cfg["data"]["train_output_dir"],
            "{}_test_pred_noCorr_2.vtp".format(filename),
        ),
        THRESH,
    )


def fit_sparse(
    slice_num, fn, chd_type, net, iter_num, two_shape_codes=False, opt_both=False
):
    STD = 0.01
    PT_NUM = 32768 // 2
    net.train()
    for param in net.parameters():
        param.requires_grad = False
    z_s = torch.normal(
        torch.zeros(
            (
                1,
                cfg["net"]["z_s_dim"],
                cfg["net"]["test_l_dim"],
                cfg["net"]["test_l_dim"],
                cfg["net"]["test_l_dim"],
            )
        ),
        std=STD,
    ).to(device)
    if two_shape_codes:
        z_s_ds = torch.normal(
            torch.zeros(
                (
                    1,
                    cfg["net"]["z_s_dim"],
                    cfg["net"]["test_l_dim"],
                    cfg["net"]["test_l_dim"],
                    cfg["net"]["test_l_dim"],
                )
            ),
            std=STD,
        ).to(device)
    else:
        z_s_ds = z_s
    z_s.requires_grad = True
    if opt_both and two_shape_codes:
        z_s_ds.requires_grad = True

    # get data
    input_data = pickle.load(open(fn, "rb"))
    seg_py = np.argmin(input_data, axis=0) + 1
    seg_py[np.all(input_data > 0.000005, axis=0)] = 0
    filename = os.path.basename(fn).split(".")[0]
    chd_type = torch.from_numpy(chd_type.astype(np.float32)).unsqueeze(0).to(device)
    # chd_type = torch.from_numpy(np.zeros_like(chd_data[i, :]).astype(np.float32)).unsqueeze(0).to(device)
    chd_type.requires_grad = False

    if net.use_diag:
        z_t = chd_type
    else:
        z_t = net.type_encoder(chd_type)

    # optimizer
    if opt_both and two_shape_codes:
        optimizer_s = torch.optim.Adam(
            [z_s, z_s_ds], lr=0.01, betas=(0.5, 0.999), weight_decay=0.0
        )
    else:
        optimizer_s = torch.optim.Adam(
            [z_s], lr=0.01, betas=(0.5, 0.999), weight_decay=0.0
        )
    optimizer_t = torch.optim.Adam(
        [chd_type], lr=0.01, betas=(0.5, 0.999), weight_decay=0.0
    )
    scheduler_s = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_s, patience=10, factor=0.5, verbose=True, min_lr=1e-6
    )
    scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_t, patience=10, factor=0.5, verbose=True, min_lr=1e-6
    )

    # optimize
    for k in range(iter_num):
        optimizer_t.zero_grad()
        optimizer_s.zero_grad()
        points, point_values = dataset.sample_slices_from_sdf(
            input_data, PT_NUM, slice_num
        )
        points = points.unsqueeze(0).to(device)
        point_values = point_values.unsqueeze(0).to(device)
        outputs = net.decoder(z_s, z_s_ds, z_t, points)

        if opt_both:
            recons_loss = torch.mean(
                ((outputs["recons"][0].squeeze(-1) - point_values) ** 2)
            )
            recons_noDs_loss = torch.mean(
                ((outputs["recons"][1].squeeze(-1) - point_values) ** 2)
                * (point_values + 1)
                * 3
            )
            gaussian_s_loss = 0.5 * torch.mean(z_s**2) + 0.5 * torch.mean(z_s_ds**2)
            total_loss = (
                1.0 * recons_loss + 1.0 * recons_noDs_loss + 0.0001 * gaussian_s_loss
            )
        else:
            recons_noDs_loss = torch.mean(
                ((outputs["recons"][1].squeeze(-1) - point_values) ** 2)
            )
            gaussian_s_loss = torch.mean(z_s**2)
            total_loss = (
                6.0 * recons_noDs_loss
                + 0.0001 * gaussian_s_loss
                + 10.0 * torch.mean(outputs["grad_mag"])
            )

        # io_utils.write_sampled_point(points, torch.sum(point_values, dim=(0,1)).unsqueeze(0), filename+'_{}.vtp'.format(k))

        total_loss.backward()
        print(k, total_loss.item())
        optimizer_s.step()
        optimizer_t.step()
        scheduler_s.step(total_loss.item())
        scheduler_t.step(total_loss.item())
    return z_s, z_s_ds, z_t, seg_py


def fit(fn, chd_type, net, iter_num, two_shape_codes=False, opt_both=False):
    STD = 0.01
    PT_NUM = 32768
    net.train()
    for param in net.parameters():
        param.requires_grad = False
    z_s = torch.normal(
        torch.zeros(
            (
                1,
                cfg["net"]["z_s_dim"],
                cfg["net"]["test_l_dim"],
                cfg["net"]["test_l_dim"],
                cfg["net"]["test_l_dim"],
            )
        ),
        std=STD,
    ).to(device)
    if two_shape_codes:
        z_s_ds = torch.normal(
            torch.zeros(
                (
                    1,
                    cfg["net"]["z_s_dim"],
                    cfg["net"]["test_l_dim"],
                    cfg["net"]["test_l_dim"],
                    cfg["net"]["test_l_dim"],
                )
            ),
            std=STD,
        ).to(device)
    else:
        z_s_ds = z_s
    z_s.requires_grad = True
    if opt_both and two_shape_codes:
        z_s_ds.requires_grad = True

    # get data
    input_data = pickle.load(open(fn, "rb"))
    seg_py = np.argmin(input_data, axis=0) + 1
    seg_py[np.all(input_data > 0.000005, axis=0)] = 0
    filename = os.path.basename(fn).split(".")[0]
    chd_type = torch.from_numpy(chd_type.astype(np.float32)).unsqueeze(0).to(device)
    # chd_type = torch.from_numpy(np.zeros_like(chd_data[i, :]).astype(np.float32)).unsqueeze(0).to(device)
    chd_type.requires_grad = False
    if net.use_diag:
        z_t = chd_type
    else:
        z_t = net.type_encoder(chd_type)

    # optimizer
    if opt_both and two_shape_codes:
        optimizer_s = torch.optim.Adam(
            [z_s, z_s_ds], lr=0.01, betas=(0.5, 0.999), weight_decay=0.0
        )
    else:
        optimizer_s = torch.optim.Adam(
            [z_s], lr=0.01, betas=(0.5, 0.999), weight_decay=0.0
        )
    optimizer_t = torch.optim.Adam(
        [chd_type], lr=0.01, betas=(0.5, 0.999), weight_decay=0.0
    )
    scheduler_s = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_s, patience=15, factor=0.6, verbose=True, min_lr=1e-6
    )
    scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_t, patience=15, factor=0.6, verbose=True, min_lr=1e-6
    )

    # optimize
    for k in range(iter_num):
        optimizer_t.zero_grad()
        optimizer_s.zero_grad()
        _, points, point_values, _ = dataset.sample_points_from_sdf(
            input_data, PT_NUM, 20, interior=False
        )
        points = points.unsqueeze(0).to(device)
        point_values = point_values.unsqueeze(0).to(device)

        outputs = net.decoder(z_s, z_s_ds, z_t, points)

        if opt_both:
            recons_loss = torch.mean(
                ((outputs["recons"][0].squeeze(-1) - point_values) ** 2)
            )
            recons_noDs_loss = torch.mean(
                ((outputs["recons"][1].squeeze(-1) - point_values) ** 2)
                * (point_values + 1)
                * 3
            )
            gaussian_s_loss = 0.5 * torch.mean(z_s**2) + 0.5 * torch.mean(z_s_ds**2)
            total_loss = (
                1.0 * recons_loss + 1.0 * recons_noDs_loss + 0.0001 * gaussian_s_loss
            )
        else:
            recons_noDs_loss = torch.mean(
                ((outputs["recons"][1].squeeze(-1) - point_values) ** 2)
            )
            gaussian_s_loss = torch.mean(z_s**2)
            total_loss = 6.0 * recons_noDs_loss + 0.0001 * gaussian_s_loss

        total_loss.backward()
        print(k, total_loss.item())
        optimizer_s.step()
        optimizer_t.step()
        scheduler_s.step(total_loss.item())
        scheduler_t.step(total_loss.item())
    return z_s, z_s_ds, z_t, seg_py


def get_motion(net, cfg, seg_dir, iter_num=200):
    STD = 0.001
    PT_NUM = 32768 // 4
    # get shape code representing motion at template space
    fns = sorted(glob.glob(os.path.join(seg_dir, "pytorch", "*.pkl")))
    num_phases = len(fns)
    mesh_fns = sorted(glob.glob(os.path.join(seg_dir, "vtk", "*.vtp")))

    # get the shape code for the first time point
    # assume healthy
    chd_type = np.zeros(len(cfg["data"]["chd_info"]["types"]))
    z_s, z_s_ds, z_t, _ = fit(
        fns[0], chd_type, net, iter_num=iter_num, opt_both=True, two_shape_codes=False
    )
    with torch.no_grad():
        sdf_noCorr = tester.z2voxel(
            z_s, z_s_ds, z_t, net, num_blocks=1, out_block=0, get_tmplt_coords=False
        )
        io_utils.write_sdf_to_vtk_mesh(
            sdf_noCorr,
            os.path.join(seg_dir, "pred_noCorr_phase{}.vtp".format(0)),
            THRESH,
        )

    with open(os.path.join(seg_dir, "z_s.pkl"), "wb") as f:
        pickle.dump(z_s, f)
    z_s.requires_grad = False
    z_t.requires_grad = False
    for i in range(len(fns)):
        mesh = load_vtk_mesh(mesh_fns[i])
        mesh = smooth_polydata(decimation(mesh, 0.5))
        points = np.flip(vtk_to_numpy(mesh.GetPoints().GetData()), axis=-1)
        points = torch.from_numpy(points.copy()).unsqueeze(0).to(device) * 2.0 - 1.0
        points_t, _, _ = net.decoder.flow(points, z_t, z_s, inverse=True)
        points_t = (np.flip(F.tanh(points_t).detach().cpu().numpy(), -1) + 1.0) / 2.0
        mesh.GetPoints().SetData(numpy_to_vtk(np.squeeze(points_t)))
        write_vtk_polydata(mesh, os.path.join(seg_dir, "tmplt_{}.vtp".format(i)))

    z_m_list = []
    z_m = None
    for i in range(num_phases):
        # for first time step, want zero displacements
        mag_wt = 10000.0 if i == 0 else 0.0
        recons_wt = 0.0 if i == 0 else 2.0
        iter_num_i = iter_num * 10 if i == 0 else iter_num * 3
        fn = fns[i]
        print(fn)
        input_data = pickle.load(open(fn, "rb"))
        seg_py = np.argmin(input_data, axis=0) + 1
        seg_py[np.all(input_data > 0.000005, axis=0)] = 0
        filename = os.path.basename(fn).split(".")[0]

        net.train()
        for param in net.parameters():
            param.requires_grad = False
        net.set_div_loss(True)
        if z_m is None:
            z_m = torch.normal(
                torch.zeros(
                    (
                        1,
                        cfg["net"]["z_s_dim"],
                        cfg["net"]["test_l_dim"],
                        cfg["net"]["test_l_dim"],
                        cfg["net"]["test_l_dim"],
                    )
                ),
                std=STD,
            ).to(device)
            # z_m = torch.normal(torch.zeros((1, cfg['net']['z_s_dim'], 3, 3, 3)), std=STD).to(device)
            z_m.requires_grad = True
            optimizer = torch.optim.Adam(
                [z_m], lr=0.01, betas=(0.5, 0.999), weight_decay=0.0
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, factor=0.5, verbose=True, min_lr=1e-4
            )
        else:
            z_m = z_m_list[-1].detach().clone()
            z_m.requires_grad = True
            optimizer = torch.optim.Adam(
                [z_m], lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, factor=0.5, verbose=True, min_lr=1e-4
            )

        for k in range(iter_num_i):

            optimizer.zero_grad()
            _, points, point_values, _ = dataset.sample_points_from_sdf(
                input_data, PT_NUM, 20, interior=False
            )
            points = points.unsqueeze(0).to(device)
            point_values = point_values.unsqueeze(0).to(device)
            points_tm, _, _ = net.decoder.flow(points, z_t, z_s, inverse=True)
            points_t, mag, total_mag = net.decoder.flow(
                F.tanh(points_tm), z_t, z_m, inverse=True
            )
            points_t_sdv = net.decoder.decoder(z_t, F.tanh(points_t))
            points_t_sdv = act(points_t_sdv).permute(0, 2, 1)

            recons_noDs_loss = torch.mean(
                ((points_t_sdv.squeeze(-1) - point_values) ** 2)
                * (point_values + 1)
                * 3
            )
            gaussian_s_loss = torch.mean(z_m**2)
            print("FLOW MAG: ", torch.mean(mag))
            total_loss = (
                recons_wt * recons_noDs_loss
                + 0.00 * gaussian_s_loss
                + 100.0 * torch.mean(mag)
                + torch.mean(total_mag) * mag_wt
            )
            # total_loss = 2.*recons_noDs_loss + 0.00 * gaussian_s_loss
            print(k, total_loss.item())

            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())
        io_utils.write_sampled_point(
            points_t, point_values, os.path.join(seg_dir, "tmplt_f_{}.vtp".format(i))
        )
        z_m.requires_grad = False
        z_m_list.append(z_m)
    # for some reason the second to last time point didn't work
    # z_m_list[-2] = 0.5 * (z_m_list[-1] + z_m_list[-3])
    with open(os.path.join(seg_dir, "motion.pkl"), "wb") as f:
        pickle.dump(z_m_list, f)


def apply_motion(net, seg_dir, cfg, type_fn=None, num_shapes=10):
    motion_fn = os.path.join(seg_dir, "motion.pkl")
    z_m_list = pickle.load(open(motion_fn, "rb"))
    if type_fn is None:
        # assume healthy here, could change to other types
        chd_type = np.zeros(len(cfg["data"]["chd_info"]["types"]))
        # chd_type = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0])
        chd_type = torch.from_numpy(chd_type.astype(np.float32)).unsqueeze(0).to(device)
        z_t = net.type_encoder(chd_type)

        type_sdf = tester.z2voxel(
            None, None, z_t, net, num_blocks=num_block, out_type=True
        )
        out_type_fn = os.path.join(seg_dir, "mesh_ori.vtp")
        type_mesh = io_utils.write_sdf_to_vtk_mesh(
            type_sdf, out_type_fn, THRESH, decimate=0.7, keep_largest=True
        )
        out_dir = seg_dir
    else:
        type_mesh = load_vtk_mesh(type_fn)
        out_dir = os.path.dirname(type_fn)
    points = np.flip(vtk_to_numpy(type_mesh.GetPoints().GetData()), axis=-1)
    points = torch.from_numpy(points.copy()).unsqueeze(0).to(device) * 2.0 - 1.0

    stats = get_shape_distribution(cfg)

    tmplt_dir = os.path.join(out_dir, os.path.basename(type_fn).split(".")[0])
    if not os.path.exists(tmplt_dir):
        os.makedirs(tmplt_dir)
    for j in range(num_shapes):
        mesh_dir = os.path.join(tmplt_dir, "mesh_{}".format(j))
        mesh_dir_less = os.path.join(tmplt_dir, "mesh_{}_less_motion".format(j))
        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)
        if not os.path.exists(mesh_dir_less):
            os.makedirs(mesh_dir_less)
        # shape might be deformed too much over time, using 0.2std
        z_s = torch.normal(
            torch.from_numpy(stats[0].astype(np.float32)),
            std=torch.from_numpy(stats[1].astype(np.float32) * 0.2),
        ).to(device)
        for i, z_m in enumerate(z_m_list):
            # original motion
            new_points_m, _, _ = net.decoder.flow(points, None, z_m, inverse=False)
            new_points_m_out = (
                np.flip(new_points_m.detach().cpu().numpy(), -1) + 1.0
            ) / 2.0
            type_mesh.GetPoints().SetData(numpy_to_vtk(np.squeeze(new_points_m_out)))
            write_vtk_polydata(
                type_mesh, os.path.join(tmplt_dir, "phase{}.vtp".format(i))
            )
            new_points, _, _ = net.decoder.flow(
                F.tanh(new_points_m), None, z_s, inverse=False
            )
            new_points = (np.flip(new_points.detach().cpu().numpy(), -1) + 1.0) / 2.0
            type_mesh.GetPoints().SetData(numpy_to_vtk(np.squeeze(new_points)))
            write_vtk_polydata(
                type_mesh, os.path.join(mesh_dir, "phase{}.vtp".format(i))
            )

            # less motion
            z_m_l = (z_m - z_m_list[0] * 0.5) + z_m_list[0]
            new_points_m, _, _ = net.decoder.flow(points, None, z_m_l, inverse=False)
            new_points, _, _ = net.decoder.flow(
                F.tanh(new_points_m), None, z_s, inverse=False
            )
            new_points = (np.flip(new_points.detach().cpu().numpy(), -1) + 1.0) / 2.0
            type_mesh.GetPoints().SetData(numpy_to_vtk(np.squeeze(new_points)))
            write_vtk_polydata(
                type_mesh, os.path.join(mesh_dir_less, "phase{}.vtp".format(i))
            )


def fit_sparse_testdata(
    slice_num,
    net,
    cfg,
    mode,
    iter_num=200,
    num_blocks=1,
    thresh=0.5,
    output_mode="sdf",
    opt_both=True,
    two_shape_codes=True,
):
    fns = sorted(glob.glob(os.path.join(cfg["data"]["test_dir"], "pytorch", "*.pkl")))
    df = dataset.read_excel(
        cfg["data"]["chd_info"]["diag_fn"],
        sheet_name=cfg["data"]["chd_info"]["diag_sn"],
    )
    fns, chd_data, _ = dataset.parse_data_by_chd_type(
        fns,
        df,
        cfg["data"]["chd_info"]["types"],
        cfg["data"]["chd_info"]["exclude_types"],
        mode=mode,
        use_aug=False,
    )
    print("Test data: ", fns, os.path.join(cfg["data"]["test_dir"], "pytorch", "*.pkl"))
    dice_list, dice_noCorr_list, diag_list = [], [], []
    for i, fn in enumerate(fns):
        filename = os.path.basename(fn).split(".")[0]
        # freeze for each sample just in case
        z_s, z_s_ds, z_t, seg_py = fit_sparse(
            slice_num,
            fn,
            chd_data[i, :],
            net,
            iter_num=iter_num,
            two_shape_codes=two_shape_codes,
            opt_both=opt_both,
        )
        with torch.no_grad():
            sdf = tester.z2voxel(
                z_s,
                z_s_ds,
                z_t,
                net,
                num_blocks=num_block,
                out_block=-1,
                get_tmplt_coords=False,
            )
            sdf_noCorr = tester.z2voxel(
                z_s,
                z_s_ds,
                z_t,
                net,
                num_blocks=num_block,
                out_block=0,
                get_tmplt_coords=False,
            )
            io_utils.write_sdf_to_vtk_mesh(
                sdf,
                os.path.join(
                    cfg["data"]["test_output_dir"], "{}_test_pred.vtp".format(filename)
                ),
                thresh,
            )
            io_utils.write_sdf_to_vtk_mesh(
                sdf_noCorr,
                os.path.join(
                    cfg["data"]["test_output_dir"],
                    "{}_test_pred_noCorr.vtp".format(filename),
                ),
                thresh,
            )
            z_s_np = np.squeeze(z_s.detach().cpu().numpy().astype(np.float32))
            np.save(
                os.path.join(
                    cfg["data"]["test_output_dir"], "{}_feat.npy".format(filename)
                ),
                z_s_np,
            )
            dice = metrics.dice_score_from_sdf(
                sdf[1:-1, 1:-1, 1:-1], seg_py, thresh=thresh
            )
            dice_noCorr = metrics.dice_score_from_sdf(
                sdf_noCorr[1:-1, 1:-1, 1:-1], seg_py, thresh=thresh
            )
            print("Unseen shape dice: ", dice)
            dice_list.append(dice)
            dice_noCorr_list.append(dice_noCorr)
    if len(dice_list) > 0:
        io_utils.write_scores(
            os.path.join(cfg["data"]["test_output_dir"], "dice_test.csv"), dice_list
        )
        io_utils.write_scores(
            os.path.join(cfg["data"]["test_output_dir"], "dice_noCorr_test.csv"),
            dice_noCorr_list,
        )


def fit_testdata(
    net,
    cfg,
    mode,
    iter_num=200,
    num_blocks=1,
    thresh=0.5,
    output_mode="sdf",
    opt_both=True,
    two_shape_codes=True,
):
    fns = sorted(glob.glob(os.path.join(cfg["data"]["test_dir"], "pytorch", "*.pkl")))
    df = dataset.read_excel(
        cfg["data"]["chd_info"]["diag_fn"],
        sheet_name=cfg["data"]["chd_info"]["diag_sn"],
    )
    fns, chd_data, _ = dataset.parse_data_by_chd_type(
        fns,
        df,
        cfg["data"]["chd_info"]["types"],
        cfg["data"]["chd_info"]["exclude_types"],
        mode=mode,
        use_aug=False,
    )
    print("Test data: ", fns, os.path.join(cfg["data"]["test_dir"], "pytorch", "*.pkl"))
    dice_list, dice_noCorr_list, diag_list = [], [], []
    for i, fn in enumerate(fns):
        filename = os.path.basename(fn).split(".")[0]
        # freeze for each sample just in case
        z_s, z_s_ds, z_t, seg_py = fit(
            fn,
            chd_data[i, :],
            net,
            iter_num=iter_num,
            two_shape_codes=two_shape_codes,
            opt_both=opt_both,
        )
        with torch.no_grad():
            sdf = tester.z2voxel(
                z_s,
                z_s_ds,
                z_t,
                net,
                num_blocks=num_block,
                out_block=-1,
                get_tmplt_coords=False,
            )
            sdf_noCorr = tester.z2voxel(
                z_s,
                z_s_ds,
                z_t,
                net,
                num_blocks=num_block,
                out_block=0,
                get_tmplt_coords=False,
            )
            io_utils.write_sdf_to_vtk_mesh(
                sdf,
                os.path.join(
                    cfg["data"]["test_output_dir"], "{}_test_pred.vtp".format(filename)
                ),
                thresh,
            )
            io_utils.write_sdf_to_vtk_mesh(
                sdf_noCorr,
                os.path.join(
                    cfg["data"]["test_output_dir"],
                    "{}_test_pred_noCorr.vtp".format(filename),
                ),
                thresh,
            )
            z_s_np = np.squeeze(z_s.detach().cpu().numpy().astype(np.float32))
            np.save(
                os.path.join(
                    cfg["data"]["test_output_dir"], "{}_feat.npy".format(filename)
                ),
                z_s_np,
            )
            dice = metrics.dice_score_from_sdf(
                sdf[1:-1, 1:-1, 1:-1], seg_py, thresh=thresh
            )
            dice_noCorr = metrics.dice_score_from_sdf(
                sdf_noCorr[1:-1, 1:-1, 1:-1], seg_py, thresh=thresh
            )
            print("Unseen shape dice: ", dice)
            dice_list.append(dice)
            dice_noCorr_list.append(dice_noCorr)
    if len(dice_list) > 0:
        io_utils.write_scores(
            os.path.join(cfg["data"]["test_output_dir"], "dice_test.csv"), dice_list
        )
        io_utils.write_scores(
            os.path.join(cfg["data"]["test_output_dir"], "dice_noCorr_test.csv"),
            dice_noCorr_list,
        )


def sample_shape_space(
    net,
    key,
    chd_type,
    cfg,
    lat_vecs=None,
    stats=None,
    num_copies=10,
    num_block=1,
    thresh=0.5,
    get_img=False,
    get_noCorr=False,
):

    filename = key

    if net.use_diag:
        z_t = torch.from_numpy(np.array([chd_type]).astype(np.float32)).to(device)
    else:
        z_t = net.type_encoder(
            torch.from_numpy(np.array([chd_type]).astype(np.float32)).to(device)
        )
    type_sdf = tester.z2voxel(None, None, z_t, net, num_blocks=num_block, out_type=True)
    out_type_fn = os.path.join(
        cfg["data"]["train_output_dir"], "{}_test_type_struct.vtp".format(filename)
    )
    type_mesh = io_utils.write_sdf_to_vtk_mesh(
        type_sdf, out_type_fn, thresh, decimate=0.7, keep_largest=True
    )
    points = np.flip(vtk_to_numpy(type_mesh.GetPoints().GetData()), axis=-1)
    points = torch.from_numpy(points.copy()).unsqueeze(0).to(device) * 2.0 - 1.0

    for i in range(num_copies):
        dx_z_s = torch.normal(
            torch.from_numpy(stats[0].astype(np.float32)),
            std=torch.from_numpy(stats[1].astype(np.float32)),
        ).to(z_t.device)
        new_points, _, _ = net.decoder.flow(points, z_t, dx_z_s, inverse=False)
        new_points = (np.flip(new_points.detach().cpu().numpy(), -1) + 1.0) / 2.0
        type_mesh.GetPoints().SetData(numpy_to_vtk(np.squeeze(new_points)))
        write_vtk_polydata(
            type_mesh,
            os.path.join(
                cfg["data"]["train_output_dir"],
                "{}_mesh_spstd0.5_r{}.vtp".format(filename, i),
            ),
        )


def interpolate_type(
    net,
    type_dict,
    curr_key,
    prev_key,
    cfg,
    interval=5,
    num_block=1,
    thresh=0.5,
    mode="interp",
):
    if net.use_diag:
        z_curr = torch.from_numpy(type_dict[curr_key].astype(np.float32)).to(device)
        z_prev = torch.from_numpy(type_dict[prev_key].astype(np.float32)).to(device)
    else:
        z_curr = net.type_encoder(
            torch.from_numpy(type_dict[curr_key].astype(np.float32)).to(device)
        )
        z_prev = net.type_encoder(
            torch.from_numpy(type_dict[prev_key].astype(np.float32)).to(device)
        )

    t_curr = torch.from_numpy(type_dict[curr_key].astype(np.float32)).to(device)
    t_prev = torch.from_numpy(type_dict[prev_key].astype(np.float32)).to(device)
    if mode == "interp":
        factor = np.linspace(0.0, 1.0, interval)
    elif mode == "extrap":
        factor = np.linspace(0.0, 1.4, interval)
    else:
        raise NameError("Undefined mode, should be interp or extrap")

    # interpolate type
    for i, f in enumerate(factor):
        if net.use_diag:
            z_t = (t_prev + (t_curr - t_prev) * f).unsqueeze(0)
        else:
            z_t = net.type_encoder(t_prev + (t_curr - t_prev) * f).unsqueeze(0)
            # z_t = (z_prev + (z_curr - z_prev) * f).unsqueeze(0)
        print("DEBUG: ", f, z_t.shape)
        type_sdf = tester.z2voxel(
            None, None, z_t, net, num_blocks=num_block, out_type=True
        )
        # io_utils.write_sdf_to_vtk_mesh(type_sdf, os.path.join(cfg['data']['train_output_dir'], '{}_to_{}_type_latentInterp{}.vtp'.format(prev_key, curr_key, i)), thresh, keep_largest=True)
        io_utils.write_sdf_to_vtk_mesh(
            type_sdf,
            os.path.join(
                cfg["data"]["train_output_dir"],
                "{}_to_{}_type_diagInterp{}.vtp".format(prev_key, curr_key, i),
            ),
            thresh,
            keep_largest=True,
        )


def interpolate_type_and_shape(
    net,
    data_curr,
    data_prev,
    cfg,
    lat_vecs=None,
    lat_vecs_ds=None,
    interval=5,
    num_block=1,
    thresh=0.5,
):
    chd_type_curr = data_curr["chd_type"].to(device)
    filename_curr = data_curr["filename"][0]
    chd_type_prev = data_prev["chd_type"].to(device)
    filename_prev = data_prev["filename"][0]
    type_curr = net.type_encoder(chd_type_curr)
    type_prev = net.type_encoder(chd_type_prev)
    shape_curr = lat_vecs(data_curr["idx"].to(device)).view(
        1,
        cfg["net"]["z_s_dim"],
        cfg["net"]["l_dim"],
        cfg["net"]["l_dim"],
        cfg["net"]["l_dim"],
    )
    shape_prev = lat_vecs(data_prev["idx"].to(device)).view(
        1,
        cfg["net"]["z_s_dim"],
        cfg["net"]["l_dim"],
        cfg["net"]["l_dim"],
        cfg["net"]["l_dim"],
    )
    shape_ds_curr = lat_vecs_ds(data_curr["idx"].to(device)).view(
        1,
        cfg["net"]["z_s_dim"],
        cfg["net"]["l_dim"],
        cfg["net"]["l_dim"],
        cfg["net"]["l_dim"],
    )
    shape_ds_prev = lat_vecs_ds(data_prev["idx"].to(device)).view(
        1,
        cfg["net"]["z_s_dim"],
        cfg["net"]["l_dim"],
        cfg["net"]["l_dim"],
        cfg["net"]["l_dim"],
    )

    factor = np.linspace(0.0, 1.0, interval)
    # interpolate type
    for i, f in enumerate(factor):
        z_t = type_prev + (type_curr - type_prev) * f
        z_s = shape_prev
        z_s_ds = shape_ds_prev
        sdf = tester.z2voxel(
            z_s,
            z_s_ds,
            z_t,
            net,
            num_blocks=num_block,
            out_block=-1,
            get_tmplt_coords=False,
        )
        sdf_noCorr = tester.z2voxel(
            z_s,
            z_s_ds,
            z_t,
            net,
            num_blocks=num_block,
            out_block=0,
            get_tmplt_coords=False,
        )
        io_utils.write_sdf_to_vtk_mesh(
            sdf,
            os.path.join(
                cfg["data"]["train_output_dir"],
                "{}_to_{}_pred_tInterp{}.vtp".format(filename_prev, filename_curr, i),
            ),
            thresh,
            keep_largest=True,
        )
        io_utils.write_sdf_to_vtk_mesh(
            sdf_noCorr,
            os.path.join(
                cfg["data"]["train_output_dir"],
                "{}_to_{}_pred_noCorr_tInterp{}.vtp".format(
                    filename_prev, filename_curr, i
                ),
            ),
            thresh,
            keep_largest=True,
        )

    # interpolate shape
    for i, f in enumerate(factor):
        z_t = type_prev
        z_s = shape_prev + (shape_curr - shape_prev) * f
        z_s_ds = shape_ds_prev + (shape_ds_curr - shape_ds_prev) * f
        sdf = tester.z2voxel(
            z_s,
            z_s_ds,
            z_t,
            net,
            num_blocks=num_block,
            out_block=-1,
            get_tmplt_coords=False,
        )
        sdf_noCorr = tester.z2voxel(
            z_s,
            z_s_ds,
            z_t,
            net,
            num_blocks=num_block,
            out_block=0,
            get_tmplt_coords=False,
        )
        io_utils.write_sdf_to_vtk_mesh(
            sdf,
            os.path.join(
                cfg["data"]["train_output_dir"],
                "{}_to_{}_pred_sInterp{}.vtp".format(filename_prev, filename_curr, i),
            ),
            thresh,
            keep_largest=True,
        )
        io_utils.write_sdf_to_vtk_mesh(
            sdf_noCorr,
            os.path.join(
                cfg["data"]["train_output_dir"],
                "{}_to_{}_pred_noCorr_sInterp{}.vtp".format(
                    filename_prev, filename_curr, i
                ),
            ),
            thresh,
            keep_largest=True,
        )


def get_shape_distribution(cfg):
    fns = glob.glob(os.path.join(cfg["data"]["train_output_dir"], "*feat.npy"))
    print("SEARCH DIR: ", os.path.join(cfg["data"]["train_output_dir"], "*feat.npy"))
    z_s_list = []
    for fn in fns:
        feat = np.load(fn)
        z_s_list.append(feat)
    z_s_list = np.array(z_s_list)
    std = np.std(np.array(z_s_list), axis=0, keepdims=True)
    mean = np.mean(np.array(z_s_list), axis=0, keepdims=True)
    print("zs shape: ", z_s_list.shape)
    print("STD: ", std)
    print("MEAN: ", mean)
    return mean, std


def get_original_prediction(
    net, z_s, z_s_ds, data, num_block=1, thresh=0.5, cfg=None, tester=None
):
    chd_type = data["chd_type"].to(device)

    z_t = chd_type if net.use_diag else net.type_encoder(chd_type)

    z_s_np = np.squeeze(z_s.detach().cpu().numpy().astype(np.float32))
    np.save(
        os.path.join(
            cfg["data"]["train_output_dir"], "{}_feat.npy".format(data["filename"][0])
        ),
        z_s_np,
    )
    z_s_ds_np = np.squeeze(z_s_ds.detach().cpu().numpy().astype(np.float32))
    np.save(
        os.path.join(
            cfg["data"]["train_output_dir"],
            "{}_ds_feat.npy".format(data["filename"][0]),
        ),
        z_s_ds_np,
    )

    curr_time = time.time()
    total_time = time.time() - curr_time

    sdf, out_points_list = tester.z2voxel(
        z_s, z_s_ds, z_t, net, num_blocks=num_block, out_block=-1, get_tmplt_coords=True
    )
    curr_time = time.time()

    sdf_noCorr = tester.z2voxel(
        z_s, z_s_ds, z_t, net, num_blocks=num_block, out_block=0, get_tmplt_coords=False
    )
    total_time += time.time() - curr_time
    print("TIME: ", total_time)

    io_utils.write_sdf_to_vtk_mesh(
        sdf_noCorr,
        os.path.join(
            cfg["data"]["train_output_dir"],
            "{}_pred_noCorr.vtp".format(data["filename"][0]),
        ),
        thresh,
    )
    sdf_diff = sdf - sdf_noCorr  # FINAL - LAST SDF
    io_utils.write_sdf_to_vtk_mesh(
        sdf,
        os.path.join(
            cfg["data"]["train_output_dir"], "{}_pred.vtp".format(data["filename"][0])
        ),
        thresh,
    )
    # io_utils.write_vtk_image(sdf_diff, os.path.join(cfg['data']['train_output_dir'], '{}_pred_ds.vti'.format(data['filename'][0])))
    dice = metrics.dice_score_from_sdf(
        sdf[1:-1, 1:-1, 1:-1], data["y"].numpy()[0], thresh=thresh
    )
    dice_noCorr = metrics.dice_score_from_sdf(
        sdf_noCorr[1:-1, 1:-1, 1:-1], data["y"].numpy()[0], thresh=thresh
    )
    print("Seen shape dice:", data["filename"][0], dice)
    z_v = z_t.cpu().detach().numpy()
    return z_v, dice, dice_noCorr, total_time


def get_type_dict(excel_info, get_new_type=False):
    # get the type of all combinations in the dataset
    df = dataset.read_excel(excel_info["diag_fn"], excel_info["diag_sn"])
    p_ids = df.index.tolist()
    all_types = df.columns.tolist()
    type_ids = [all_types.index(t) for t in excel_info["types"]]
    all_types = [all_types[t] for t in type_ids]
    print("Allowed types:  ", excel_info["types"])
    arr = df.to_numpy()[:, type_ids]
    print(arr.shape)
    type_dict = {}
    for r in arr:
        type_name = "_".join(itertools.compress(all_types, r == 1))
        type_name = "Normal" if type_name == "" else type_name
        if type_name not in type_dict.keys():
            type_dict[type_name] = r
    print("Existing types: ", type_dict.keys())
    if get_new_type:
        # get new combinations of types not in the dataset
        for i in range(len(excel_info["types"])):
            ts_combs = list(itertools.combinations(excel_info["types"], i + 1))
            for ts in ts_combs:
                type_name = "_".join(ts)
                type_name = "Normal" if type_name == "" else type_name
                if (type_name not in type_dict.keys()) and (
                    type_name + "_NE" not in type_dict.keys()
                ):
                    type_values = np.zeros(len(excel_info["types"]))
                    for t in ts:
                        type_values[ts.index(t)] = 1.0
                    type_dict[type_name + "_NE"] = type_values
    print("Existing types: ", type_dict.keys())
    return type_dict


def test_type_prediction(net, excel_info, thresh=0.5):
    type_dict = get_type_dict(excel_info, get_new_type=True)
    for k in type_dict.keys():
        if net.use_diag:
            z_t = torch.from_numpy(np.array([type_dict[k]]).astype(np.float32)).to(
                device
            )
        else:
            z_t = net.type_encoder(
                torch.from_numpy(np.array([type_dict[k]]).astype(np.float32)).to(device)
            )
        print("SHAPE: ", z_t.shape)
        type_sdf = tester.z2voxel(
            None, None, z_t, net, num_blocks=num_block, out_type=True
        )
        print(np.min(type_sdf), np.max(type_sdf))
        io_utils.write_sdf_to_vtk_mesh(
            type_sdf,
            os.path.join(
                cfg["data"]["train_output_dir"], "{}_exist_type.vtp".format(k)
            ),
            thresh,
            keep_largest=True,
        )


def save_type_net(net, path):
    checkpoint = {}
    if not net.use_diag:
        t_enc = net.type_encoder
        if isinstance(t_enc, torch.nn.DataParallel):
            checkpoint["t_enc"] = t_enc.module.state_dict()
        else:
            checkpoint["t_enc"] = t_enc.state_dict()
    t_dec = net.decoder.decoder
    if isinstance(t_dec, torch.nn.DataParallel):
        checkpoint["t_dec"] = t_dec.module.state_dict()
    else:
        checkpoint["t_dec"] = t_dec.state_dict()
    torch.save(checkpoint, path)


def test_invertibility(net, cfg, mode):
    if not os.path.exists(os.path.join(cfg["data"]["test_output_dir"], "backward")):
        os.makedirs(os.path.join(cfg["data"]["test_output_dir"], "backward"))
    if not os.path.exists(os.path.join(cfg["data"]["test_output_dir"], "forward")):
        os.makedirs(os.path.join(cfg["data"]["test_output_dir"], "forward"))
    if not os.path.exists(os.path.join(cfg["data"]["test_output_dir"], "original")):
        os.makedirs(os.path.join(cfg["data"]["test_output_dir"], "original"))

    fns = sorted(glob.glob(os.path.join(cfg["data"]["test_dir"], "pytorch", "*.pkl")))
    df = dataset.read_excel(
        cfg["data"]["chd_info"]["diag_fn"],
        sheet_name=cfg["data"]["chd_info"]["diag_sn"],
    )
    fns, chd_data = dataset.parse_data_by_chd_type(
        fns, df, cfg["data"]["chd_info"]["types"], mode=mode, use_aug=False
    )
    for fn in fns:
        filename = os.path.basename(fn).split(".")[0]
        z_s_np = np.load(
            os.path.join(cfg["data"]["test_output_dir"], "{}_feat.npy".format(filename))
        )
        mesh = load_vtk_mesh(
            os.path.join(
                cfg["data"]["test_output_dir"],
                "{}_test_pred_noCorr.vtp".format(filename),
            )
        )
        mesh = smooth_polydata(decimation(mesh, 0.5))
        z_s = torch.from_numpy(z_s_np).to(device).unsqueeze(0)
        points = np.flip(vtk_to_numpy(mesh.GetPoints().GetData()), axis=-1)
        points = torch.from_numpy(points.copy()).unsqueeze(0).to(device) * 2.0 - 1.0

        type_points, _, _ = net.decoder.flow(points, None, z_s, inverse=True)

        new_points, _, _ = net.decoder.flow(type_points, None, z_s, inverse=False)

        type_mesh = vtk.vtkPolyData()
        type_mesh.DeepCopy(mesh)
        type_points = (np.flip(type_points.detach().cpu().numpy(), -1) + 1.0) / 2.0
        type_mesh.GetPoints().SetData(numpy_to_vtk(np.squeeze(type_points)))
        new_mesh = vtk.vtkPolyData()
        new_mesh.DeepCopy(mesh)
        new_points = (np.flip(new_points.detach().cpu().numpy(), -1) + 1.0) / 2.0
        new_mesh.GetPoints().SetData(numpy_to_vtk(np.squeeze(new_points)))
        write_vtk_polydata(
            type_mesh,
            os.path.join(
                cfg["data"]["test_output_dir"],
                "backward",
                "{}_test_pred_noCorr.vtp".format(filename),
            ),
        )
        write_vtk_polydata(
            new_mesh,
            os.path.join(
                cfg["data"]["test_output_dir"],
                "forward",
                "{}_test_pred_noCorr.vtp".format(filename),
            ),
        )
        write_vtk_polydata(
            mesh,
            os.path.join(
                cfg["data"]["test_output_dir"],
                "original",
                "{}_test_pred_noCorr.vtp".format(filename),
            ),
        )


def image_mesh_synthesize(
    net, lat_vecs, stats, num_copies=1, num_block=1, thresh=0.5, cfg=None, tester=None
):
    mesh_dir = os.path.join(cfg["data"]["train_dir"], "vtk")
    train = dataset.ImgSDFDataset(
        cfg["data"]["train_dir"],
        cfg["train"]["n_smpl_pts"],
        chd_info=cfg["data"]["chd_info"],
        mode=["train"],
        use_aug=False,
    )
    dataloader = DataLoader(train, batch_size=1, shuffle=False, pin_memory=True)
    syn_out_dir = os.path.join(cfg["data"]["output_dir"], "synthesize")
    if not os.path.exists(syn_out_dir):
        os.makedirs(os.path.join(syn_out_dir, "mesh"))
        os.makedirs(os.path.join(syn_out_dir, "image"))
        os.makedirs(os.path.join(syn_out_dir, "seg"))

    for i, data in enumerate(dataloader):
        filename = data["filename"][0]
        if os.path.exists(
            os.path.join(
                syn_out_dir, "image", filename + "_{}.vti".format(num_copies - 1)
            )
        ):
            continue
        mesh_fn = os.path.join(mesh_dir, filename + ".vtp")
        mesh = load_vtk_mesh(mesh_fn)
        mesh = smooth_polydata(decimation(mesh, 0.5))
        img = data["image"].to(device)
        print("original debug range: ", torch.min(img), torch.max(img), torch.mean(img))
        z_s = lat_vecs(data["idx"].to(device)).view(
            1,
            cfg["net"]["z_s_dim"],
            cfg["net"]["l_dim"],
            cfg["net"]["l_dim"],
            cfg["net"]["l_dim"],
        )
        points = np.flip(vtk_to_numpy(mesh.GetPoints().GetData()), axis=-1)
        points = torch.from_numpy(points.copy()).unsqueeze(0).to(device) * 2.0 - 1.0
        type_points, _, _ = net.decoder.flow(points, None, z_s, inverse=True)

        chd_type = data["chd_type"].to(device)
        if not net.use_diag:
            z_t = net.type_encoder(chd_type)
        else:
            z_t = chd_type

        for j in range(num_copies):
            z_s_new = torch.normal(
                torch.from_numpy(stats[0].astype(np.float32)),
                std=torch.from_numpy(stats[1].astype(np.float32) * 0.5),
            ).to(device)
            # z_s_new = torch.normal(torch.zeros_like(z_s), std=torch.ones_like(z_s)*0.005).to(device)
            new_points, _, _ = net.decoder.flow(
                type_points, None, z_s_new, inverse=False
            )
            new_mesh = vtk.vtkPolyData()
            new_mesh.DeepCopy(mesh)
            new_points = (np.flip(new_points.detach().cpu().numpy(), -1) + 1.0) / 2.0
            new_mesh.GetPoints().SetData(numpy_to_vtk(np.squeeze(new_points)))
            new_image = tester.deform_image(
                z_s, z_s_new, z_t, img, net, num_blocks=1, order=1
            )
            print(new_image.shape)
            print(
                "debug range: ",
                np.min(new_image[1:-1, 1:-1, 1:-1, :]),
                np.max(new_image[1:-1, 1:-1, 1:-1, :]),
                np.mean(new_image[1:-1, 1:-1, 1:-1, :]),
            )
            write_vtk_polydata(
                new_mesh,
                os.path.join(syn_out_dir, "mesh", filename + "_{}.vtp".format(j)),
            )
            io_utils.write_vtk_image(
                new_image[1:-1, 1:-1, 1:-1, :],
                os.path.join(syn_out_dir, "image", filename + "_{}.vti".format(j)),
            )


def get_vsd_variations(cfg, mode, net, code, vsd_id, num_var=10, thresh=0.5):
    train = dataset.SDFDataset(
        cfg["data"]["train_dir"],
        cfg["train"]["n_smpl_pts"],
        cfg["data"]["output_dir"],
        chd_info=cfg["data"]["chd_info"],
        mode=MODE,
        use_aug=False,
        pad_num=0,
    )
    dataloader_train = DataLoader(train, batch_size=1, shuffle=False, pin_memory=True)
    with torch.no_grad():
        for i, data in enumerate(dataloader_train):
            z_s = lat_vecs(data["idx"].to(device)).view(
                1,
                cfg["net"]["z_s_dim"],
                cfg["net"]["l_dim"],
                cfg["net"]["l_dim"],
                cfg["net"]["l_dim"],
            )
            for k, vsd in enumerate(torch.linspace(0, 1, num_var)):
                chd_type = data["chd_type"].to(device)
                chd_type[:, vsd_id] = vsd
                z_t = chd_type if net.use_diag else net.type_encoder(chd_type)
                sdf_noCorr = tester.z2voxel(
                    z_s,
                    z_s,
                    z_t,
                    net,
                    num_blocks=1,
                    out_block=0,
                    get_tmplt_coords=False,
                )
                io_utils.write_sdf_to_vtk_mesh(
                    sdf_noCorr,
                    os.path.join(
                        cfg["data"]["train_output_dir"],
                        "{}_pred_noCorr_vsdVar{}.vtp".format(data["filename"][0], k),
                    ),
                    thresh,
                )


def random_type_generation(cfg, net, num_gen, thresh=0.5):
    out_dir = os.path.join(cfg["data"]["train_output_dir"], "random_type_gen")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    type_dict = get_type_dict(cfg["data"]["chd_info"], get_new_type=False)
    stats = get_shape_distribution(cfg)
    keys = list(type_dict.keys())
    type_list = []
    for i in range(num_gen):
        k_id = random.choice(keys)
        type_i = type_dict[k_id]
        type_list.append(type_i)
        if net.use_diag:
            z_t = torch.from_numpy(np.array([type_i]).astype(np.float32)).to(device)
        else:
            z_t = net.type_encoder(
                torch.from_numpy(np.array([type_i]).astype(np.float32)).to(device)
            )
        z_s = torch.normal(
            torch.from_numpy(stats[0].astype(np.float32)),
            std=torch.from_numpy(stats[1].astype(np.float32)),
        ).to(z_t.device)
        sdf_noCorr = tester.z2voxel(
            z_s, z_s, z_t, net, num_blocks=1, out_block=0, get_tmplt_coords=False
        )
        io_utils.write_sdf_to_vtk_mesh(
            sdf_noCorr, os.path.join(out_dir, "gen_{}.vtp".format(i)), thresh
        )
    io_utils.write_scores(os.path.join(out_dir, "type_gt.csv"), type_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--config")
    parser.add_argument("--grid_size", type=int, default=2)
    args = parser.parse_args()

    MODE = ["train"]
    num_block = 1
    with open(args.config, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)

    test_ops = cfg["test_ops"]
    if not os.path.exists(cfg["data"]["output_dir"]):
        os.makedirs(cfg["data"]["output_dir"])

    THRESH = 0.5
    tester = SDF4CHDTester(
        device,
        cell_grid_size=args.grid_size,
        out_dim=cfg["net"]["out_dim"],
        sampling_threshold=THRESH,
        binary=True if cfg["train"]["binary"] else False,
    )
    dice_score_list, dice_noCorr_score_list, time_list = [], [], []
    z_vector_list = {}

    # TRAINING ACCURACY
    train = dataset.SDFDataset(
        cfg["data"]["train_dir"],
        cfg["train"]["n_smpl_pts"],
        cfg["data"]["output_dir"],
        chd_info=cfg["data"]["chd_info"],
        mode=MODE,
        use_aug=False,
        pad_num=cfg["train"]["pad_num"],
    )
    dataloader_test = DataLoader(train, batch_size=1, shuffle=False, pin_memory=True)
    # create network and latent codes
    net = SDF4CHD(
        in_dim=0,
        out_dim=cfg["net"]["out_dim"],
        num_types=len(cfg["data"]["chd_info"]["types"]),
        z_t_dim=cfg["net"]["z_t_dim"],
        z_s_dim=cfg["net"]["z_s_dim"],
        type_mlp_num=cfg["net"]["type_mlp_num"],
        ds_mlp_num=cfg["net"]["ds_mlp_num"],
        dx_mlp_num=cfg["net"]["dx_mlp_num"],
        latent_dim=cfg["net"]["latent_dim"],
        ins_norm=cfg["net"]["ins_norm"],
        type_bias=False,
        lip_reg=cfg["net"]["lip_reg"],
        step_size=cfg["net"]["step_size"],
        use_diag=cfg["net"]["use_diag"],
        act_func=net_utils.act if cfg["train"]["binary"] else lambda x: x,
    )
    # initialize Z_s

    lat_vecs = torch.nn.Embedding(
        len(train.idx_dict),
        cfg["net"]["z_s_dim"]
        * cfg["net"]["l_dim"]
        * cfg["net"]["l_dim"]
        * cfg["net"]["l_dim"],
        max_norm=1.0,
    ).to(device)
    # lat_vecs.load_state_dict(torch.load(os.path.join(cfg['data']['output_dir'], 'code{}.pt'.format(args.epoch)), map_location=torch.device('cpu'))['latent_codes'])
    if cfg["net"]["two_shape_codes"]:
        lat_vecs_ds = torch.nn.Embedding(
            len(train.idx_dict),
            cfg["net"]["z_s_dim"]
            * cfg["net"]["l_dim"]
            * cfg["net"]["l_dim"]
            * cfg["net"]["l_dim"],
            max_norm=1.0,
        ).to(device)
        # lat_vecs_ds.load_state_dict(torch.load(os.path.join(cfg['data']['output_dir'], 'code_ds{}.pt'.format(args.epoch)), map_location=torch.device('cpu'))['latent_codes'])
    else:
        lat_vecs_ds = lat_vecs

    net.to(device)
    net.load_state_dict(
        torch.load(
            os.path.join(cfg["data"]["output_dir"], "net_{}.pt".format(args.epoch)),
            map_location=torch.device("cpu"),
        )["state_dict"]
    )
    # save type network for segmentation
    save_type_net(
        net,
        os.path.join(cfg["data"]["output_dir"], "type_net_{}.pt".format(args.epoch)),
    )

    cfg["data"]["train_output_dir"] = os.path.join(
        cfg["data"]["output_dir"], MODE[0] + "_{}".format(args.epoch)
    )
    if not os.path.exists(cfg["data"]["train_output_dir"]):
        os.makedirs(cfg["data"]["train_output_dir"])

    if test_ops["type_pred"]:
        # TEST 1: TYPE PREDICTION
        test_type_prediction(net, cfg["data"]["chd_info"], thresh=THRESH)

    if test_ops["type_interp"]:
        # TEST 2: TYPE INTERPOLATION
        type_dict = get_type_dict(cfg["data"]["chd_info"], get_new_type=False)
        ## For all
        interp_groups = [["VSD_ToF", "VSD_TGA"], ["Normal", "VSD_PuA"]]
        # interp_groups = [['VSD', 'SV_VSD'], ['Normal', 'SV_VSD']]
        for g in interp_groups:
            interpolate_type(
                net,
                type_dict,
                g[1],
                g[0],
                cfg,
                interval=21,
                num_block=1,
                thresh=THRESH,
                mode="extrap",
            )
        interp_groups = [
            ["VSD_ToF", "VSD_DORV"],
            ["VSD_DORV", "VSD_TGA"],
            ["Normal", "VSD_ToF"],
            ["VSD_ToF", "VSD_PuA"],
        ]
        for g in interp_groups:
            interpolate_type(
                net, type_dict, g[1], g[0], cfg, interval=11, num_block=1, thresh=THRESH
            )

    if test_ops["train_dice"]:
        # TEST 3: WH DICE ACCURACY ON SEEN SHAPES
        with torch.no_grad():
            for i, data in enumerate(dataloader_test):
                original_copy = bool(
                    re.match("ct_[a-z]+_\d+", data["filename"][0])
                ) or bool(re.match("ct_\d+_image", data["filename"][0]))
                print(i, data["filename"][0], original_copy)
                if original_copy:
                    z_s = lat_vecs(data["idx"].to(device)).view(
                        1,
                        cfg["net"]["z_s_dim"],
                        cfg["net"]["l_dim"],
                        cfg["net"]["l_dim"],
                        cfg["net"]["l_dim"],
                    )
                    if cfg["net"]["two_shape_codes"]:
                        z_s_ds = lat_vecs_ds(data["idx"].to(device)).view(
                            1,
                            cfg["net"]["z_s_dim"],
                            cfg["net"]["l_dim"],
                            cfg["net"]["l_dim"],
                            cfg["net"]["l_dim"],
                        )
                        z_vector, dice, dice_noCorr, total_time = (
                            get_original_prediction(
                                net,
                                z_s,
                                z_s_ds,
                                data,
                                num_block=num_block,
                                thresh=THRESH,
                                cfg=cfg,
                                tester=tester,
                            )
                        )
                    else:
                        z_vector, dice, dice_noCorr, total_time = (
                            get_original_prediction(
                                net,
                                z_s,
                                z_s,
                                data,
                                num_block=num_block,
                                thresh=THRESH,
                                cfg=cfg,
                                tester=tester,
                            )
                        )
                    dice_score_list.append(dice)
                    dice_noCorr_score_list.append(dice_noCorr)
                    time_list.append(total_time)
            print("TIME AVG: ", np.mean(np.array(total_time)))
        io_utils.write_scores(
            os.path.join(cfg["data"]["train_output_dir"], "dice.csv"), dice_score_list
        )
        io_utils.write_scores(
            os.path.join(cfg["data"]["train_output_dir"], "dice_noCorr.csv"),
            dice_noCorr_score_list,
        )
        print(dice_score_list)

    if test_ops["vsd_variation"]:
        get_vsd_variations(
            cfg,
            ["train"],
            net,
            lat_vecs,
            vsd_id=cfg["data"]["chd_info"]["types"].index("VSD"),
            num_var=5,
            thresh=0.5,
        )

    if test_ops["test_dice"]:
        # TEST 4: WH DICE ACCURACY ON UNSEEN SHAPES
        MODE = ["test"]
        # MODE = ['validate']
        cfg["data"]["test_output_dir"] = os.path.join(
            cfg["data"]["output_dir"],
            MODE[0]
            + "_both_lat{}_fixType_opt_{}".format(cfg["net"]["test_l_dim"], args.epoch),
        )
        if not os.path.exists(cfg["data"]["test_output_dir"]):
            os.makedirs(cfg["data"]["test_output_dir"])
        fit_testdata(
            net,
            cfg,
            mode=MODE,
            iter_num=260,
            thresh=THRESH,
            opt_both=True,
            two_shape_codes=cfg["net"]["two_shape_codes"],
        )

    if test_ops["shape_type_interp"]:
        # TEST 5: TYPE AND SHAPE INTERPOLATION
        MODE = ["train"]
        selected_list = ["ct_1017_image", "ct_1012_image"]
        data_dict = OrderedDict()
        for i, data in enumerate(dataloader_test):
            original_copy = bool(
                re.match("ct_[a-z]+_\d+", data["filename"][0])
            ) or bool(re.match("ct_\d+_image", data["filename"][0]))
            # if original_copy and data['filename'][0] in selected_list:
            print(data["filename"][0], original_copy)
            if original_copy and data["filename"][0] in selected_list:
                data_dict[data["filename"][0]] = data
        with torch.no_grad():
            data_prev = None
            for key in selected_list:
                data = data_dict[key]
                if data_prev is not None:
                    interpolate_type_and_shape(
                        net,
                        data,
                        data_prev,
                        cfg,
                        lat_vecs=lat_vecs,
                        lat_vecs_ds=lat_vecs_ds,
                        interval=21,
                        num_block=num_block,
                        thresh=THRESH,
                    )
                data_prev = data
    if test_ops["shape_gen"]:
        # TEST 6: SHAPE GENERATION
        # assert test_ops['train_dice'], "Need to run training data to estimate distribution before generation"
        stats = get_shape_distribution(cfg)
        type_dict = get_type_dict(cfg["data"]["chd_info"], get_new_type=False)
        with torch.no_grad():
            for i, (key, chd_type) in enumerate(type_dict.items()):
                print(key)
                sample_shape_space(
                    net,
                    key,
                    chd_type,
                    cfg,
                    lat_vecs=lat_vecs,
                    stats=stats,
                    num_copies=30,
                    num_block=num_block,
                    thresh=THRESH,
                    get_img=False,
                    get_noCorr=False,
                )

    if test_ops["invertible"]:
        MODE = ["test"]
        cfg["data"]["test_output_dir"] = os.path.join(
            cfg["data"]["output_dir"],
            MODE[0]
            + "_both_lat{}_fixType_opt_{}".format(cfg["net"]["test_l_dim"], args.epoch),
        )
        test_invertibility(net, cfg, mode=MODE)

    if test_ops["rand_type_gen"]:
        random_type_generation(cfg, net, num_gen=50)

    if test_ops["rand_motion_gen"]:
        template_dir = cfg["data"]["template_mesh_dir"]
        seg_dir = cfg["data"]["motion_segmentation_dir"]
        mesh_fns = glob.glob(os.path.join(template_dir, "*.vtp"))

        if not os.path.exists(os.path.join(seg_dir, "motion.pkl")):
            get_motion(net, cfg, seg_dir, iter_num=10)
        for f in mesh_fns:
            print(f)
            apply_motion(net, seg_dir, cfg, type_fn=f, num_shapes=10)
