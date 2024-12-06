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
import io_utils

# ['ASD', 'VSD', 'AVSD', 'Normal']
class ImgSDFDataset(Dataset):
    def __init__(self, root_dir, chd_info=None, mode='train', use_aug=True):
        if isinstance(root_dir, str):
            self.seg_fns = sorted(glob.glob(os.path.join(root_dir, 'pytorch', '*.pkl')))
            self.im_fns = sorted(glob.glob(os.path.join(root_dir, 'pytorch_img', '*.pkl')))
        else:
            self.seg_fns, self.im_fns = [], []
            for r in root_dir:
                self.seg_fns += sorted(glob.glob(os.path.join(r, 'pytorch', '*.pkl')))
                self.im_fns += sorted(glob.glob(os.path.join(r, 'pytorch_img', '*.pkl')))
        self.root_dir = root_dir

        self.diag_data = None
        if chd_info is not None:
            df = read_excel(chd_info['diag_fn'], sheet_name=chd_info['diag_sn'])
            self.seg_fns, self.diag_data, self.idx_dict = parse_data_by_chd_type(self.seg_fns, df, chd_info['types'], chd_info['exclude_types'], mode=mode, use_aug=use_aug)
            self.im_fns, _, _ = parse_data_by_chd_type(self.im_fns, df, chd_info['types'], chd_info['exclude_types'], mode=mode, use_aug=use_aug)
        for i, j  in zip(self.seg_fns, self.im_fns):
            assert os.path.basename(i) == os.path.basename(j), "Unmatched segmentation and image data in the training dataset"
    
    def __len__(self):
        return len(self.im_fns)

    def get_file_name(self, item):
        return self.im_fns[item]

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        if len(self.seg_fns) > 0:  # check if there are ground truths to compare to
            sdf_py = pickle.load(open(self.seg_fns[item], 'rb'))
            # convert to segmentation
            # if sdf values are greater than zero, assign to background
            seg_py = np.argmin(sdf_py, axis=0)+1
            seg_py[np.all(sdf_py>0.000005, axis=0)] = 0
        
            sdf_py = (sdf_py < 0.000005).astype(np.float32) # C H W D
            sdf_py = torch.from_numpy(sdf_py.astype(np.float32))

        img_file_name = self.im_fns[item]

        img_py = pickle.load(open(img_file_name, 'rb'))
        
        # if not normalized
        if np.max(img_py) > 100:
            img_py = img_py -  np.min(img_py)
            img_py = np.clip(img_py, 0., 2000.)/2000.
        elif np.min(img_py) < 0.: # this is for syn data
            img_py = img_py -  np.min(img_py)
            img_py /= np.max(img_py)

        if len(self.seg_fns) > 0:
            data_dict = {'image': torch.from_numpy(img_py.astype(np.float32)).unsqueeze(0), \
                'y': torch.from_numpy(seg_py.astype(np.uint8)), \
                'filename': os.path.basename(self.seg_fns[item]).split('.')[0], \
                'processed_sdf': sdf_py}
        else:  # if only predicting (no ground truth)
            data_dict = {'image': torch.from_numpy(img_py.astype(np.float32)).unsqueeze(0), \
                    'filename': os.path.basename(self.im_fns[item]).split('.')[0]}
        return data_dict

class ImgNiftyDataset(Dataset):
    def __init__(self, root_dir, ext='*.nii.gz'):
        print(os.path.join(root_dir, ext))
        if isinstance(root_dir, str):
            self.im_fns = sorted(glob.glob(os.path.join(root_dir, ext)))
        else:
            self.im_fns = []
            for r in root_dir:
                self.im_fns += sorted(glob.glob(os.path.join(r, ext)))
        self.root_dir = root_dir
        self.ext = ext
    
    def __len__(self):
        return len(self.im_fns)

    def get_file_name(self, item):
        return self.im_fns[item]

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        
        img_fn = self.im_fns[item]
        img_py = io_utils.import_nifty_img(img_fn, size=(128,128,128), order=1)
    
        # if not normalized
        if np.max(img_py) > 100 and np.max(img_py) < 10000:
            img_py = img_py -  np.min(img_py)
            img_py = np.clip(img_py, 0., 2000.)/2000.
        elif np.min(img_py) < 0.: # this is for syn data
            img_py = img_py -  np.min(img_py)
            img_py /= np.max(img_py)
        elif np.max(img_py) > 10000: #this is for mr - to-do
            upper = np.percentile(img_py, 90)
            print("MR:", upper)
            img_py = np.clip(img_py, 0., upper)/upper

        data_dict = {'image': torch.from_numpy(img_py.astype(np.float32)).unsqueeze(0), \
                    'img_fn': img_fn}
        return data_dict
