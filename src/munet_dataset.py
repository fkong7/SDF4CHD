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
from dataset import parse_data_by_chd_type, read_excel

# ['ASD', 'VSD', 'AVSD', 'Normal']
class ImgSDFDataset(Dataset):
    def __init__(self, root_dir, chd_info=None, mode='train', use_aug=True):
        self.seg_fns = sorted(glob.glob(os.path.join(root_dir, 'pytorch', '*.pkl')))
        self.im_fns = sorted(glob.glob(os.path.join(root_dir, 'pytorch_img', '*.pkl')))
        self.root_dir = root_dir

        self.diag_data = None
        if chd_info is not None:
            df = read_excel(chd_info['diag_fn'], sheet_name=chd_info['diag_sn'])
            self.seg_fns, self.diag_data = parse_data_by_chd_type(self.seg_fns, df, chd_info['types'], mode=mode, use_aug=use_aug)
            self.im_fns, _ = parse_data_by_chd_type(self.im_fns, df, chd_info['types'], mode=mode, use_aug=use_aug)
    
    def __len__(self):
        return len(self.seg_fns)

    def get_file_name(self, item):
        return self.seg_fns[item]

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        sdf_py = pickle.load(open(self.seg_fns[item], 'rb'))
        # convert to segmentation
        # if sdf values are greater than zero, assign to background
        seg_py = np.argmin(sdf_py, axis=0)+1
        seg_py[np.all(sdf_py>0.000005, axis=0)] = 0
        
        sdf_py = (sdf_py < 0.000005).astype(np.float32) # C H W D
        sdf_py = torch.from_numpy(sdf_py.astype(np.float32))

        img_file_name = self.im_fns[item]

        img_py = pickle.load(open(img_file_name, 'rb'))
        img_py = img_py -  np.min(img_py)
        img_py = np.clip(img_py, 0., 2000.)/2000.

        data_dict = {'image': torch.from_numpy(img_py.astype(np.float32)).unsqueeze(0), \
                'y': torch.from_numpy(seg_py.astype(np.uint8)), \
                'filename': os.path.basename(self.seg_fns[item]).split('.')[0], \
                'processed_sdf': sdf_py}
        return data_dict
