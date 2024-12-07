import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../vtk_utils'))                       
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from pre_process import resample_spacing
from vtk_utils import *
import vtk 
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, get_vtk_array_type
import numpy as np
import SimpleITK as sitk                                                                    
import pickle 
import torch
import glob
import h5py
import warnings
warnings.filterwarnings("ignore")
vtk.vtkObject.GlobalWarningDisplayOff()

try:
    from mpi4py import MPI
except Exception as e: print(e)

def convert_vtk_to_trimesh(mesh):
    coords = vtk_to_numpy(mesh.GetPoints().GetData())
    cells = vtk_to_numpy(mesh.GetPolys().GetData())
    cells = cells.reshape(mesh.GetNumberOfCells(), 4)
    cells = cells[:,1:]
    return trimesh.Trimesh(vertices=coords, faces=(cells), process=False)

def convert_mesh_to_sign_distance_map(mesh, name='SignedDistances', img_info=None):
    '''
    Convert a VTK polygon mesh to a sign distance volume:
    Args:
        mesh: vtkPolyData
        name: name of the point data array to store the sign distance values
        img_info: dict of image information, spacing, min_bound(origin), size
    Returns:
        img: vtkImageData of the sign distance volume
    '''
    if img_info is None:
        img_info = {}
        extra = 0.2
        size = [64, 64, 64]
        coords = vtk_to_numpy(mesh.GetPoints().GetData())
        min_bound, max_bound = np.min(coords, axis=0), np.max(coords, axis=0)
        extend = max_bound - min_bound
        min_bound -= extend * extra/2.
        img_info['spacing'] = extend * (1.+extra) / np.array(size)
        img_info['min_bound'] = min_bound
        img_info['size'] = size

    img = vtk.vtkImageData()
    img.SetDimensions(img_info['size'])
    img.SetSpacing(img_info['spacing'])
    img.SetOrigin(img_info['min_bound'])
    implicitPolyDataDistance = vtk.vtkImplicitPolyDataDistance()
    implicitPolyDataDistance.SetInput(mesh)
    signedDistances = vtk.vtkFloatArray()
    signedDistances.SetNumberOfComponents(1)
    signedDistances.SetName(name)

    for i in range(img_info['size'][0]):
        for j in range(img_info['size'][1]):
            for k in range(img_info['size'][2]):
                physical_coord = [0., 0., 0.]
                img.TransformContinuousIndexToPhysicalPoint([k, j, i], physical_coord)
                signedDistance = implicitPolyDataDistance.EvaluateFunction(physical_coord)
                signedDistances.InsertNextValue(signedDistance)
    img.GetPointData().SetScalars(signedDistances)
    return img

def crop_to_structure(img, seg, seg_ids, relax_vox=5):
    seg_py = sitk.GetArrayFromImage(seg).transpose(2, 1, 0)
    x, y, z = seg_py.shape

    max_x_a, max_y_a, max_z_a = -1, -1, -1
    min_x_a, min_y_a, min_z_a = 1000, 1000, 1000
    for seg_id in seg_ids:
        ids_x, ids_y, ids_z = np.where(seg_py==seg_id)
        max_x, max_y, max_z = min(np.max(ids_x)+relax_vox, x-1), min(np.max(ids_y)+relax_vox, y-1), min(np.max(ids_z)+relax_vox, z-1)
        min_x, min_y, min_z = max(np.min(ids_x)-relax_vox, 0), max(np.min(ids_y)-relax_vox, 0), max(np.min(ids_z)-relax_vox, 0)
        max_x_a = max_x if max_x > max_x_a else max_x_a
        max_y_a = max_y if max_y > max_y_a else max_y_a
        max_z_a = max_z if max_z > max_z_a else max_z_a
      
        min_x_a = min_x if min_x < min_x_a else min_x_a
        min_y_a = min_y if min_y < min_y_a else min_y_a
        min_z_a = min_z if min_z < min_z_a else min_z_a
    return img[min_x_a:max_x_a, min_y_a:max_y_a, min_z_a:max_z_a], seg[min_x_a:max_x_a, min_y_a:max_y_a, min_z_a:max_z_a]

def create(seg_fn, hs_size, ds_size, ref_fn=None, r_ids=[1]):
    seg = sitk.ReadImage(seg_fn)
    if ref_fn is not None:
        ref = sitk.ReadImage(ref_fn)
        seg = sitk.Resample(seg, ref.GetSize(),
                                 sitk.Transform(),
                                 sitk.sitkNearestNeighbor,
                                 ref.GetOrigin(),
                                 ref.GetSpacing(),
                                 ref.GetDirection(),
                                 0,
                                 seg.GetPixelID())
    seg_hs = resample_spacing(seg, template_size=hs_size, order=0)[0]
    seg_hs.SetSpacing(np.ones(3)/np.array(hs_size))
    seg_vtk = exportSitk2VTK(seg_hs, seg_hs.GetSpacing())[0]
    
    seg_ds = resample_spacing(seg, template_size=ds_size, order=0)[0]
    seg_ds.SetSpacing(np.ones(3)/np.array(ds_size))
    seg_ds_vtk = exportSitk2VTK(seg_ds, seg_ds.GetSpacing())[0]
    
    mesh = vtk_marching_cube_multi(seg_vtk, 0)
    img_info = {'spacing': seg_ds.GetSpacing(), 'min_bound': seg_ds_vtk.GetOrigin(), 'size': seg_ds_vtk.GetDimensions()}
    region_ids = np.unique(vtk_to_numpy(mesh.GetCellData().GetArray(0))).astype(int)
    # Check if all ids are present
    if not set(r_ids).issubset(set(region_ids)):
        return None, None, None
    mesh.GetCellData().GetArray(0).SetName('Scalars_')
    
    mesh_list, py_sdf = [], []
    sdf = vtk.vtkImageData()
    sdf.SetDimensions(img_info['size'])
    sdf.SetSpacing(img_info['spacing'])
    sdf.SetOrigin(img_info['min_bound'])
    
    for i in r_ids: # Myo only
        mesh_i = thresholdPolyData(mesh, 'Scalars_', (i, i), 'cell')
        mesh_i = fill_hole(mesh_i)
        mesh_i = get_largest_connected_polydata(mesh_i)
        mesh_i = smooth_polydata(mesh_i, 25, smoothingFactor=0.5)
        #mesh_i = decimation(mesh_i, 0.8)
        mesh_i = smooth_polydata(mesh_i, 5, smoothingFactor=0.5)
        sdf_i = convert_mesh_to_sign_distance_map(mesh_i, 'sdf_{}'.format(i), img_info)
        if sdf is None:
            sdf = sdf_i
        else:
            sdf.GetPointData().AddArray(sdf_i.GetPointData().GetArray('sdf_{}'.format(i)))
        x, y, z = sdf.GetDimensions()
        py_sdf_i = vtk_to_numpy(sdf.GetPointData().GetArray('sdf_{}'.format(i))).reshape(z, y, x).transpose(2, 1, 0).astype(np.float32)
        py_sdf.append(py_sdf_i)
        mesh_list.append(mesh_i)
    return appendPolyData(mesh_list), sdf, np.array(py_sdf)



def resample_image(source, target, order=1):
    if order==1:
        interp = sitk.sitkLinear
    else:
        interp = sitk.sitkNearestNeighbor
    source = sitk.Resample(source, target.GetSize(),
                             sitk.Transform(),
                             interp,
                             target.GetOrigin(),
                             target.GetSpacing(),
                             target.GetDirection(),
                             0,
                             source.GetPixelID())
    return source

def median_filter(seg, k_size):
    seg_py = sitk.GetArrayFromImage(seg)
    seg_py[:, :, 0] = 0
    seg_py[:, :, -1] = 0
    seg_py[:, 0, :] = 0
    seg_py[:, -1, :] = 0
    seg_py[0, :, :] = 0
    seg_py[-1, :, :] = 0
    seg_new = sitk.GetImageFromArray(seg_py)
    seg_new.CopyInformation(seg)
    filt = sitk.MedianImageFilter()
    filt.SetRadius(k_size)
    out = filt.Execute(seg_new)
    return out 

def closing_filter(seg, k_size):
    seg_py = sitk.GetArrayFromImage(seg)
    seg_py[:, :, 0] = 0
    seg_py[:, :, -1] = 0
    seg_py[:, 0, :] = 0
    seg_py[:, -1, :] = 0
    seg_py[0, :, :] = 0
    seg_py[-1, :, :] = 0
    filt = sitk.BinaryMorphologicalClosingImageFilter()
    filt.SetKernelRadius(k_size)

    seg_py_new = np.zeros_like(seg_py).astype(seg_py.dtype)
    for i in np.unique(seg_py):
        if i==0:
            continue
        tmp = sitk.GetImageFromArray((seg_py==i).astype(seg_py.dtype))
        tmp_c = filt.Execute(tmp)
        tmp_c_py = sitk.GetArrayFromImage(tmp_c)
        seg_py_new[tmp_c_py==1] = i
    seg_new = sitk.GetImageFromArray(seg_py_new)
    seg_new.CopyInformation(seg)
    return seg_new

def create_from_segmentation(seg_fn, hs_size, ds_size, ref_fn=None, r_ids=[1]):
    seg = sitk.ReadImage(seg_fn)
    if ref_fn is not None:
        ref = sitk.ReadImage(ref_fn)
        seg = resample_image(seg, ref, order=0)
    seg_hs = resample_spacing(seg, template_size=hs_size, order=0)[0]
    seg_hs.SetSpacing(np.ones(3)/np.array(hs_size))
    
    seg_ds = resample_spacing(seg, template_size=ds_size, order=0)[0]
    seg_ds.SetSpacing(np.ones(3)/np.array(ds_size))
    
    img_info = {'spacing': seg_ds.GetSpacing(), 'min_bound': seg_ds.GetOrigin(), 'size': seg_ds.GetSize()}
    sdf = vtk.vtkImageData()
    sdf.SetDimensions(img_info['size'])
    sdf.SetSpacing(img_info['spacing'])
    sdf.SetOrigin(img_info['min_bound'])
    
    mesh_list, mesh_ls_list, py_sdf = [], [], []
    region_ids = np.array([])
    for i in r_ids: # Myo only
        seg_py = sitk.GetArrayFromImage(seg_hs)
        seg_i = sitk.GetImageFromArray((seg_py==i).astype(np.uint8))
        seg_i.CopyInformation(seg_hs)
        seg_i_vtk = exportSitk2VTK(seg_i, seg_i.GetSpacing())[0]
        mesh_i = smooth_polydata(vtk_marching_cube(seg_i_vtk, 0, 1), 20, smoothingFactor=0.5)
        mesh_i = decimation(mesh_i, 0.9)
        print("MESH: ", i, mesh_i.GetNumberOfPoints())
        region_ids = np.append(region_ids, np.ones(mesh_i.GetNumberOfPoints()) * (i-1))
        distance = sitk.SignedMaurerDistanceMapImageFilter()
        distance.InsideIsPositiveOff()
        distance.UseImageSpacingOn()
        # write sdf to vti
        sdf_i = distance.Execute(seg_i)
        sdf_i = resample_image(sdf_i, seg_ds, order=1)
        sdf_i = exportSitk2VTK(sdf_i, sdf_i.GetSpacing())[0]
        # create 0 level set to visualize GT
        mesh_ls_i = vtk_marching_cube_continuous(sdf_i, 0.000005)
        mesh_ls_list.append(mesh_ls_i)

        sdf_i.GetPointData().GetArray(0).SetName('sdf_{}'.format(i))
        if sdf is None:
            sdf = sdf_i
        else:
            sdf.GetPointData().AddArray(sdf_i.GetPointData().GetArray('sdf_{}'.format(i)))
        x, y, z = sdf.GetDimensions()
        py_sdf_i = vtk_to_numpy(sdf.GetPointData().GetArray('sdf_{}'.format(i))).reshape(z, y, x).transpose(2, 1, 0).astype(np.float32)
        py_sdf.append(py_sdf_i)
        mesh_list.append(mesh_i)
    mesh = appendPolyData(mesh_list)
    mesh_ls = appendPolyData(mesh_ls_list)
    region_ids_arr = numpy_to_vtk(region_ids)
    region_ids_arr.SetName('RegionId')
    mesh.GetPointData().AddArray(region_ids_arr)
    return mesh, mesh_ls, sdf, np.array(py_sdf)
   
def process_img(img, ref, ds_size):
    img = sitk.Resample(img, ref.GetSize(),
                             sitk.Transform(),
                             sitk.sitkLinear,
                             ref.GetOrigin(),
                             ref.GetSpacing(),
                             ref.GetDirection(),
                             0,
                             img.GetPixelID())
    img_ds = resample_spacing(img, template_size=ds_size, order=1)[0]
    img_ds.SetSpacing(np.ones(3)/np.array(ds_size))
    img_ds_vtk = exportSitk2VTK(img_ds, img_ds.GetSpacing())[0]
    x, y, z = img_ds_vtk.GetDimensions()
    py_img = vtk_to_numpy(img_ds_vtk.GetPointData().GetScalars()).reshape(z, y, x).transpose(2, 1, 0).astype(np.float32)
    img_ds_vtk.GetPointData().SetScalars(numpy_to_vtk(py_img.transpose(2, 1, 0).flatten()))
    return img_ds_vtk, py_img

def check_image_mesh_alignment(img, mesh):
    x, y, z = img.GetDimensions()
    py_img = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(z, y, x).transpose(2, 1, 0)
    coords = vtk_to_numpy(mesh.GetPoints().GetData())
    coords *= np.expand_dims(np.array(py_img.shape), 0)
    coords = coords.astype(int)
    py_img[coords[:, 0], coords[:, 1], coords[:, 2]] = 1000.
    py_img = py_img.transpose(2,1,0).reshape(z*y*z)
    img.GetPointData().SetScalars(numpy_to_vtk(py_img))
    return img
    
def partition_sdf_into_chunks(sdf, target_size):
    size = sdf.shape
    x = size[1] // target_size[0]
    y = size[2] // target_size[1]
    z = size[3] // target_size[2]
    out_dict = {}
    x_list = np.array_split(sdf, x, axis=1)
    x_coord_list = np.array_split(np.arange(size[1]), x)
    count = 0
    for x_sdf, x_coord in zip(x_list, x_coord_list):
        # record the index (coordinates) for each chunk to add back to coords
        x_i = x_coord[0]
        y_list = np.array_split(x_sdf, y, axis=2)
        y_coord_list = np.array_split(np.arange(size[2]), y)
        for xy_sdf, y_coord in zip(y_list, y_coord_list):
            y_i = y_coord[0]
            z_list = np.array_split(xy_sdf, z, axis=3)
            z_coord_list = np.array_split(np.arange(size[3]), z)
            for xyz_sdf, z_coord in zip(z_list, z_coord_list):
                z_i = z_coord[0]
                if np.sum(xyz_sdf<0) > 100:
                    out_dict[str(count)] = {'sdf_chunk': xyz_sdf, \
                        'chunk_coord': np.array([x_i, y_i, z_i]), 'total_size': sdf.shape}
                    count += 1
    return out_dict


if __name__ == '__main__':

    img_dir = '/scratch/users/fwkong/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final_Apr11/alter5Joint1000_latent4_lipx100_NOuseDiag_Init1017_DivMag0.01_Pad0_GradMag0_smplfac20_twophase/train_2200/shape_gen/img_large'
    img_dir = None
    seg_dir = 'data/segmentations'
    out_dir = 'data/processed_segs'

    r_ids=[1, 2, 3, 4, 5, 6, 7]
    ref_fn = None

    folder_list = ['vtk', 'vtk_ls', 'pytorch', 'vtk_img', 'pytorch_img'] 
    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        comm_size = comm.Get_size()
    except Exception as e:
        print(e)
        comm = None
        comm_size = 1
        rank = 0
    
    if rank ==0:
        for f in folder_list:
            if not os.path.exists(os.path.join(out_dir, f)):
                os.makedirs(os.path.join(out_dir, f))
        # TO-DO change this
        seg_fns = sorted(glob.glob(os.path.join(seg_dir, '*.nii.gz')))
        seg_fns_scatter = [None] * comm_size
        chunck_size = len(seg_fns) // comm_size
        for i in range(comm_size-1):
            seg_fns_scatter[i] = seg_fns[i*chunck_size:(i+1)*chunck_size]
        seg_fns_scatter[-1] = seg_fns[(comm_size-1)*chunck_size:]
    else:
        seg_fns_scatter = None
    if comm is not None:
        seg_fns_scatter = comm.scatter(seg_fns_scatter, root=0)
    else:
        seg_fns_scatter = seg_fns

    for seg_fn in seg_fns_scatter:
        if img_dir:
            img_fn = os.path.join(img_dir, os.path.basename(seg_fn))
        name = os.path.basename(seg_fn).split('.nii.gz')[0]
        mesh, mesh_ls, sdf_v, sdf_v_py = create_from_segmentation(seg_fn, (512, 512, 512), (128, 128, 128), ref_fn, r_ids)
        if not mesh:
            continue
        write_vtk_polydata(mesh, os.path.join(out_dir, 'vtk', '{}.vtp'.format(name)))
        pickle.dump(sdf_v_py.astype(np.float32),open(os.path.join(out_dir, 'pytorch', '{}.pkl'.format(name)), 'wb'))
        if img_dir:
            img_v, img_v_py = process_img(sitk.ReadImage(img_fn), sitk.ReadImage(seg_fn), (128, 128, 128))
            write_vtk_image(sdf_v, os.path.join(out_dir, 'vtk', '{}.vti'.format(name)))
            pickle.dump(img_v_py.astype(np.float32),open(os.path.join(out_dir, 'pytorch_img', '{}.pkl'.format(name)), 'wb'))

