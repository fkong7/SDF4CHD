import os
import torch
from torch.utils.data import Dataset
from vtk_utils.vtk_utils import *
import torch.nn.functional as F
import pickle
import numpy as np
import glob
import pandas as pd
import h5py
import random

def read_excel(fn, sheet_name="Sheet1"):
    df = pd.read_excel(fn, sheet_name=sheet_name, header=0, index_col=1, engine='openpyxl')
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    df.drop(index=df.index[0], axis=0, inplace=True)
    df = df.fillna(0)
    return df

def parse_data_by_chd_type(fns, df, type_names, mode=['train'], use_aug=True):
    arr = df.to_numpy()
    # check sum, if no diagnosis, set to normal
    patient_ids = df.index.tolist()
    all_types = df.columns.tolist()
    # find all files with the specified chd types
    type_ids = [all_types.index(t) for t in type_names]
    # Append training, validation or testing mode
    mask = arr[:, 0] > -1
    for m in mode:
        mode_id = all_types.index(m)
        mask2 = arr[:, mode_id] > 0
        mask = np.logical_and(mask, mask2)
    #mask = np.logical_and(mask, mask2)
    ids_to_keep = np.array(patient_ids)[mask].astype(int)

    fns_to_keep = []
    type_data = []
    for fn in fns:
        # do not use augmentation data for validation or testing
        if (not use_aug) and 'image_' in os.path.basename(fn):
            pass
        else:
            basename = os.path.basename(fn)
            for p_id in ids_to_keep:
                if str(int(p_id)) in basename:
                    fns_to_keep.append(fn)
                    type_data.append(arr[patient_ids.index(p_id), type_ids])
    print(ids_to_keep)
    return fns_to_keep, np.array(type_data)

# device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
def sample_points_from_sdf(tmplt, n_pt, factor=5, chunk_coord=None, total_size=None):
    img_binary_py = (tmplt<0.).astype(np.float32) # C H W D
    _, m, l, n = tmplt.shape
    
    # Additive probability over all classes. Higher prob if close to the surfaces of more classes
    prob_total = np.zeros(m*l*n)
    for i in range(tmplt.shape[0]):
        prob = (np.max(np.abs(tmplt[i])) - np.abs(tmplt[i]).flatten()) # zero has the highest probability
        prob = np.exp(prob/np.max(prob)*factor)
        prob /= np.sum(prob)
        prob_total += prob
    prob_total /= np.sum(prob_total)

    # select points based on sampling probability
    select = np.random.choice(np.arange(m*l*n, dtype=np.int64), n_pt, p=prob_total, replace=False)
    x = (select // (l*n)).astype(np.float32)
    y = ((select - x * l * n) // n).astype(np.float32)
    z = (select - x * l * n - y*n).astype(np.float32)
    #print(x, y, z)
    x += np.random.normal(0., 1./3., n_pt).astype(np.float32)
    y += np.random.normal(0., 1./3., n_pt).astype(np.float32)
    z += np.random.normal(0., 1./3., n_pt).astype(np.float32)
    #print(x, y, z)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    z = torch.from_numpy(z)
    # normalize
    x_nrm = 2.*(x.float() / float(tmplt.shape[1]) - 0.5)
    y_nrm = 2.*(y.float() / float(tmplt.shape[2]) - 0.5)
    z_nrm = 2.*(z.float() / float(tmplt.shape[3]) - 0.5)
    if chunk_coord is not None:
        x_nrm_total = 2.*((x.float() + float(chunk_coord[0])) / float(total_size[1]) - 0.5)
        y_nrm_total = 2.*((y.float() + float(chunk_coord[1])) / float(total_size[2]) - 0.5)
        z_nrm_total = 2.*((z.float() + float(chunk_coord[2])) / float(total_size[3]) - 0.5)
        points_total = torch.stack([z_nrm_total, y_nrm_total, x_nrm_total], dim=-1)
    
    points = torch.stack([z_nrm, y_nrm, x_nrm], dim=-1)
    img_binary = torch.from_numpy(img_binary_py)
    points_gs = points.unsqueeze(0).unsqueeze(0).unsqueeze(0) #(1, 1, 1, N, 3)
    point_values = F.grid_sample(img_binary.unsqueeze(0), points_gs, padding_mode='border', align_corners=True)  # (C, 1, 1, N)
    point_values = point_values.squeeze(2).squeeze(2).squeeze(0)
    if chunk_coord is not None:
        return img_binary, points_total, point_values
    else:
        return img_binary, points, point_values

def sample_points_from_mesh(mesh, n_pts):
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    # uniform probability
    select = np.random.choice(np.arange(mesh.GetNumberOfPoints(), dtype=np.int64), n_pts, replace=False)

    sampled_points = points[select, :]
    sampled_points = sampled_points + np.random.normal(scale=0.03, size=sampled_points.shape)

    sampled_points_vtk = vtk.vtkPoints()
    sampled_points_vtk.SetData(numpy_to_vtk(sampled_points))
    sampled_points_poly = vtk.vtkPolyData()
    sampled_points_poly.SetPoints(sampled_points_vtk)
    verts = vtk.vtkVertexGlyphFilter()
    verts.AddInputData(sampled_points_poly)
    verts.Update()
    dist = vtk.vtkDistancePolyDataFilter()
    dist.SetSignedDistance(True)
    #dist.SetNegateDistance(True)
    dist.SetComputeSecondDistance(False)
    dist.SetInputData(0, verts.GetOutput())
    dist.SetInputData(1, mesh)
    dist.Update()
    
    point_values = vtk_to_numpy(dist.GetOutput().GetPointData().GetArray(0))
    point_values_binary = (point_values<0.).astype(np.float32)
    sampled_points = np.flip(sampled_points, axis=-1)
    return torch.from_numpy(sampled_points.astype(np.float32)), torch.from_numpy(point_values_binary)

class ImgSDFDataset(Dataset):
    def __init__(self, root_dir, n_pts, factor=20, chd_info=None, mode='train', use_aug=True):
        self.fns = sorted(glob.glob(os.path.join(root_dir, 'pytorch', '*.pkl')))
        self.im_fns = sorted(glob.glob(os.path.join(root_dir, 'pytorch_img', '*.pkl')))
        self.root_dir = root_dir
        self.n_pts = n_pts
        self.factor = factor
        self.mode = mode

        self.diag_data = None
        if chd_info is not None:
            df = read_excel(chd_info['diag_fn'], sheet_name=chd_info['diag_sn'])
            self.fns, self.diag_data = parse_data_by_chd_type(self.fns, df, chd_info['types'], mode=mode, use_aug=use_aug)
            self.im_fns, _ = parse_data_by_chd_type(self.im_fns, df, chd_info['types'], mode=mode, use_aug=use_aug)
        self._archives = None

    def __len__(self):
        return len(self.fns)

    def get_file_name(self, item):
        return self.fns[item]

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        sdf_py_total = pickle.load(open(self.fns[item], 'rb'))
        seg_py = np.argmin(sdf_py_total, axis=0)+1
        seg_py[np.all(sdf_py_total>0., axis=0)] = 0
        gt_binary, points, point_values = sample_points_from_sdf(sdf_py_total, self.n_pts, self.factor)

        img_file_name = self.im_fns[item]

        img_py = pickle.load(open(img_file_name, 'rb'))
        img_py = np.clip(img_py, 0., 2000.)/2000.
            
        data_dict = {'image': torch.from_numpy(img_py.astype(np.float32)).unsqueeze(0), \
                'points': points, 'pt_sdv': point_values.squeeze(), \
                'filename': os.path.basename(self.fns[item]).split('.')[0], 'y': seg_py, 'gt_binary':gt_binary, 'sdf': sdf_py_total}
        if self.diag_data is not None:
            type_data = self.diag_data[item, :]
            data_dict['chd_type'] = torch.from_numpy(type_data.astype(np.float32))
        return data_dict

class SDFDataset(Dataset):
    def __init__(self, root_dir, n_pts, factor=20, chd_info=None, mode='train', use_aug=True):
        self.fns = sorted(glob.glob(os.path.join(root_dir, 'pytorch', '*.pkl')))
        self.root_dir = root_dir
        self.n_pts = n_pts
        self.factor = factor
        self.mode = mode

        self.diag_data = None
        if chd_info is not None:
            df = read_excel(chd_info['diag_fn'], sheet_name=chd_info['diag_sn'])
            self.fns, self.diag_data = parse_data_by_chd_type(self.fns, df, chd_info['types'], mode=mode, use_aug=use_aug)

    def __len__(self):
        return len(self.fns)

    def get_file_name(self, item):
        return self.fns[item]

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        sdf_py_total = pickle.load(open(self.fns[item], 'rb'))
        seg_py = np.argmin(sdf_py_total, axis=0)+1
        seg_py[np.all(sdf_py_total>0., axis=0)] = 0
        gt_binary, points, point_values = sample_points_from_sdf(sdf_py_total, self.n_pts, self.factor)

        data_dict = {'idx': item, 'points': points, 'pt_sdv': point_values.squeeze(), \
                'filename': os.path.basename(self.fns[item]).split('.')[0], 'y': seg_py, 'gt_binary':gt_binary, 'sdf': sdf_py_total}
        if self.diag_data is not None:
            type_data = self.diag_data[item, :]
            data_dict['chd_type'] = torch.from_numpy(type_data.astype(np.float32))
        return data_dict

class MeshDataset(Dataset):
    def __init__(self, root_dir, n_pts):
        self.fns = sorted(glob.glob(os.path.join(root_dir, 'pytorch', '*.pkl')))
        self.mesh_fns = sorted(glob.glob(os.path.join(root_dir, 'vtk', '*.vtp')))
        self.root_dir = root_dir
        self.n_pts = n_pts

    def __len__(self):
        return len(self.fns)

    def get_file_name(self, item):
        return self.fns[item]

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        file_name = self.fns[item]
        mesh_file_name = self.mesh_fns[item]
        mesh = load_vtk_mesh(mesh_file_name)
        points, point_values = sample_points_from_mesh(mesh, self.n_pts)
        data_dict = {'idx': item, 'points': points, 'pt_sdv': point_values.squeeze(), \
                'filename': os.path.basename(self.fns[item]).split('.')[0], 'y': seg_py, 'gt_binary':gt_binary, 'sdf': sdf_py_total}
        if self.diag_data is not None:
            type_data = self.diag_data[item, :]
            data_dict['chd_type'] = torch.from_numpy(type_data.astype(np.float32))
        return data_dict


