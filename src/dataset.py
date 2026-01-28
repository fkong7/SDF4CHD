import os
import torch
from torch.utils.data import Dataset
import io_utils
import torch.nn.functional as F
import pickle
import numpy as np
import glob
import pandas as pd
import h5py
import random
import SimpleITK as sitk
import re


def read_excel(fn, sheet_name="Sheet1"):
    df = pd.read_excel(
        fn, sheet_name=sheet_name, header=0, index_col=1, engine="openpyxl"
    )
    df.drop(columns=df.columns[0], inplace=True)
    df = df[df.index.notnull()]
    df = df.fillna(0)
    return df


def pad_types(arr, patient_ids, type_ids, type_names, pad_num=20):
    # find normal
    normal_ids = np.where(np.sum(arr[:, type_ids], axis=-1) == 0)[0]
    all_add_ids = np.array([]).astype(int)
    if len(normal_ids) < pad_num:
        print("Padding normal: ", pad_num - len(normal_ids))
        add_ids = np.random.choice(
            patient_ids[normal_ids], pad_num - len(normal_ids), replace=True
        )
        all_add_ids = np.concatenate((all_add_ids, add_ids))
    # Find other types
    for i, t in enumerate(type_ids):
        ids = np.where(arr[:, t] == 1)[0]
        if len(ids) < pad_num:
            print("Padding {}: ".format(type_names[i]), pad_num - len(ids))
            add_ids = np.random.choice(
                patient_ids[ids], pad_num - len(ids), replace=True
            )
            all_add_ids = np.concatenate((all_add_ids, add_ids))
    return np.concatenate((patient_ids, all_add_ids))


def parse_data_by_chd_type(
    fns,
    df,
    type_names,
    exclude_type_names,
    mode=["train"],
    use_aug=True,
    pad_num=0,
    ext=".pkl",
):
    arr = df.to_numpy()
    patient_ids = df.index.tolist()
    all_types = df.columns.tolist()
    # Append training, validation or testing mode
    # initialize as false
    mask = arr[:, 0] < -1
    for m in mode:
        mode_id = all_types.index(m)
        mask2 = arr[:, mode_id] > 0
        mask = np.logical_or(mask, mask2)
    ids_to_keep = np.array(patient_ids)[mask]

    type_ids = [all_types.index(t) for t in type_names]
    # if a patient has a diagnosis outside of type_names, remove
    exclude_type_ids = [
        all_types.index(t) if t != "Normal" else -1 for t in exclude_type_names
    ]
    for i in exclude_type_ids:
        # Handle normal here.
        if i == -1:
            mask[np.sum(arr[:, type_ids], axis=-1) == 0] = False
        else:
            mask[arr[:, i] > 0.0] = False
    ids_to_keep = np.array(patient_ids)[mask]
    if pad_num > 0:
        masked_arr = arr[mask, :]
        ids_to_keep = pad_types(
            masked_arr, ids_to_keep, type_ids, type_names, pad_num=pad_num
        )
    fns_to_keep = []
    type_data = []
    # need to use the same idx for padded points, store in dict
    idx_dict = {}
    for fn in fns:
        # do not use augmentation data for validation or testing
        basename = os.path.basename(fn)
        original_copy = (
            bool(re.match("ct_[a-z]+_\d+" + ext, basename))
            or bool(re.match("ct_\d+_image" + ext, basename))
            or bool(re.match(".*_mesh_.*" + ext, basename))
            or bool(re.match(".*_vsdVar.*.pkl", basename))
        )
        aug_copy = bool(re.match("ct_[a-z]+_\d+_[0-6]" + ext, basename)) or bool(
            re.match("ct_\d+_image_[0-6]" + ext, basename)
        )
        if original_copy or (use_aug and aug_copy):
            ## TO-DO not optimal, temporary for segmentation only
            # if bool(re.match(".*_mesh_.*.pkl", basename)):
            #    fns_to_keep.append(fn)
            # if bool(re.match(".*_vsdVar.*.pkl", basename)):
            #    fns_to_keep.append(fn)
            for p_id in ids_to_keep:
                if type(p_id) == float or type(p_id) == np.float64:
                    p_id = int(p_id)
                if str(p_id) in basename:
                    fns_to_keep.append(fn)
                    type_data.append(arr[patient_ids.index(p_id), type_ids])
                    if not fn in idx_dict:
                        idx_dict[fn] = len(idx_dict)
    return fns_to_keep, np.array(type_data), idx_dict


def sample_points_from_sdf(
    tmplt,
    n_pt,
    factor=5,
    chunk_coord=None,
    total_size=None,
    interior=False,
    binary=True,
):
    _, m, l, n = tmplt.shape

    # Additive probability over all classes. Higher prob if close to the surfaces of more classes
    prob_total = np.zeros((m, l, n))
    for i in range(tmplt.shape[0]):
        prob = np.where(
            tmplt[i] < 0.0,
            tmplt[i] / np.max(tmplt[i]) * np.min(tmplt[i]) * -1.0,
            tmplt[i],
        )
        prob = np.max(np.abs(prob)) - np.abs(prob)  # zero has the highest probability
        if np.mean(prob) == 0.0:
            prob = np.zeros_like(prob)
        else:
            prob = np.exp(prob / np.mean(prob) * factor)
            prob /= np.sum(prob)
        prob_total += prob

    # half of the points are on the boundary, the rest sampled from the prob distribution
    if interior:
        include = tmplt < 0.0
    else:
        include = np.abs(tmplt) < 1e-3
    x, y, z = np.where(np.any(include, axis=0))
    prob = np.sum(include, axis=0).astype(np.float32)
    prob = prob[np.any(include, axis=0)]
    prob /= np.sum(prob)
    if interior:
        select = np.random.choice(
            np.arange(len(x), dtype=np.int64), n_pt, p=prob.flatten(), replace=True
        )
    else:
        select = np.random.choice(
            np.arange(len(x), dtype=np.int64),
            (n_pt * 3) // 4,
            p=prob.flatten(),
            replace=True,
        )
    x, y, z = x[select], y[select], z[select]
    prob_out = prob.flatten()[select]
    if not interior:
        # select points based on sampling probability
        x_c, y_c, z_c = np.where(prob_total > np.percentile(prob_total, 0.97))
        prob_selected = prob_total[x_c, y_c, z_c]
        prob_selected /= np.sum(prob_selected)
        select = np.random.choice(
            np.arange(len(x_c), dtype=np.int64),
            n_pt - (n_pt * 3) // 4,
            p=prob_selected,
            replace=False,
        )
        x = np.concatenate([x, x_c[select]])
        y = np.concatenate([y, y_c[select]])
        z = np.concatenate([z, z_c[select]])
    x, y, z = x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)
    x += np.random.normal(0.0, 1.0, n_pt).astype(np.float32)
    y += np.random.normal(0.0, 1.0, n_pt).astype(np.float32)
    z += np.random.normal(0.0, 1.0, n_pt).astype(np.float32)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    z = torch.from_numpy(z)
    # normalize
    x_nrm = 2.0 * (x.float() / float(tmplt.shape[1]) - 0.5)
    y_nrm = 2.0 * (y.float() / float(tmplt.shape[2]) - 0.5)
    z_nrm = 2.0 * (z.float() / float(tmplt.shape[3]) - 0.5)

    points = torch.stack([z_nrm, y_nrm, x_nrm], dim=-1)
    points_gs = points.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, N, 3)
    if binary:
        img_binary_py = (tmplt < 0.000005).astype(np.float32)  # C H W D
        img_binary = torch.from_numpy(img_binary_py)
        point_values_binary = F.grid_sample(
            img_binary.unsqueeze(0),
            points_gs,
            padding_mode="border",
            align_corners=True,
        )  # (C, 1, 1, N)
        point_values_binary = point_values_binary.squeeze(2).squeeze(2).squeeze(0)
    img_sdv = torch.from_numpy(tmplt.astype(np.float32))
    point_values_sdv = F.grid_sample(
        img_sdv.unsqueeze(0), points_gs, padding_mode="border", align_corners=True
    )  # (C, 1, 1, N)
    point_values_sdv = point_values_sdv.squeeze(2).squeeze(2).squeeze(0)
    if binary:
        return img_binary, points, point_values_binary, point_values_sdv
    else:
        return (
            img_sdv,
            points,
            torch.clamp(point_values_sdv, min=-0.001, max=0.001),
            point_values_sdv,
        )


class SDFDataset(Dataset):
    def __init__(
        self,
        root_dir,
        n_pts,
        type_dir,
        factor=20,
        chd_info=None,
        mode=["train"],
        use_aug=True,
        use_cf=False,
        use_error=True,
        select_fn_list=None,
        train=False,
        pad_num=0,
        binary=True,
    ):
        self.fns = sorted(glob.glob(os.path.join(root_dir, "pytorch", "*.pkl")))
        self.root_dir = root_dir
        self.n_pts = n_pts
        self.factor = factor
        self.mode = mode
        self.type_dir = type_dir
        self.use_cf = use_cf
        self.use_error = use_error
        self.train = train
        self.binary = binary

        self.diag_data = None
        if select_fn_list is not None:
            select_fns = []
            for k in select_fn_list:
                for fn in self.fns:
                    if k in fn:
                        select_fns.append(fn)
            self.fns = select_fns
            print("Selected files are: ", self.fns)

        if chd_info is not None:
            df = read_excel(chd_info["diag_fn"], sheet_name=chd_info["diag_sn"])
            self.fns, self.diag_data, self.idx_dict = parse_data_by_chd_type(
                self.fns,
                df,
                chd_info["types"],
                chd_info["exclude_types"],
                mode=mode,
                use_aug=use_aug,
                pad_num=pad_num,
            )
            print(self.fns)

    def __len__(self):
        return len(self.fns)

    def get_file_name(self, item):
        return self.fns[item]

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        sdf_py_total = pickle.load(open(self.fns[item], "rb"))
        seg_py = np.argmin(sdf_py_total, axis=0) + 1
        seg_py[np.all(sdf_py_total > 0.0, axis=0)] = 0
        gt_binary, points, point_values, point_values_sdv = sample_points_from_sdf(
            sdf_py_total, self.n_pts, self.factor, binary=self.binary
        )

        if self.diag_data is not None:
            type_data = self.diag_data[item, :]
            data_dict["chd_type"] = torch.from_numpy(type_data.astype(np.float32))
            key = "".join(str(int(i)) for i in list(type_data))
        return data_dict
