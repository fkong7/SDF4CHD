import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'vtk_utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from pre_process import resample_spacing
from vtk_utils import *
import vtk 
import SimpleITK as sitk                                                                    
import pickle 
import torch
import glob
import h5py
#from mesh_to_sdf import mesh_to_sdf
#import trimesh
try:
    from mpi4py import MPI
except Exception as e: print(e)

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

def create_from_segmentation(seg_fn, hs_size, ds_size, ref_fn=None, r_ids=[1]):
    seg = sitk.ReadImage(seg_fn)
    if ref_fn is not None:
        ref = sitk.ReadImage(ref_fn)
        seg = resample_image(seg, ref, order=0)
    # crop to the Myo
    #img, seg = crop_to_structure(img, seg, r_ids, 15)
    seg_hs = resample_spacing(seg, template_size=hs_size, order=0)[0]
    seg_hs.SetSpacing(np.ones(3)/np.array(hs_size))
    
    seg_ds = resample_spacing(seg, template_size=ds_size, order=0)[0]
    seg_ds.SetSpacing(np.ones(3)/np.array(ds_size))
    
    img_info = {'spacing': seg_ds.GetSpacing(), 'min_bound': seg_ds.GetOrigin(), 'size': seg_ds.GetSize()}
    sdf = vtk.vtkImageData()
    sdf.SetDimensions(img_info['size'])
    sdf.SetSpacing(img_info['spacing'])
    sdf.SetOrigin(img_info['min_bound'])
    
    mesh_list, py_sdf = [], []
    region_ids = np.array([])
    for i in r_ids: # Myo only
        print("R_id: ", i)
        seg_py = sitk.GetArrayFromImage(seg_hs)
        seg_i = sitk.GetImageFromArray((seg_py==i).astype(np.uint8))
        seg_i.CopyInformation(seg_hs)
        seg_i_vtk = exportSitk2VTK(seg_i, seg_i.GetSpacing())[0]
        mesh_i = decimation(smooth_polydata(vtk_marching_cube(seg_i_vtk, 0, 1), 20, smoothingFactor=0.5), 0.9)
        region_ids = np.append(region_ids, np.ones(mesh_i.GetNumberOfPoints()) * (i-1))
        distance = sitk.SignedMaurerDistanceMapImageFilter()
        distance.InsideIsPositiveOff()
        distance.UseImageSpacingOn()
        sdf_i = distance.Execute(seg_i)
        sdf_i = resample_image(sdf_i, seg_ds, order=1)
        sdf_i = exportSitk2VTK(sdf_i, sdf_i.GetSpacing())[0]
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
    region_ids_arr = numpy_to_vtk(region_ids)
    region_ids_arr.SetName('RegionId')
    mesh.GetPointData().AddArray(region_ids_arr)
    return mesh, sdf, np.array(py_sdf)
   
if __name__ == '__main__':

    seg_dir ='/Users/fanweikong/Documents/Modeling/CHD/datasets/imageCHDcleaned/masks_all'
    out_dir = os.path.join(seg_dir, 'imageCHDcleaned_all')
    ref_fn = None
    r_ids=[1, 2, 3, 4, 5, 6, 7]

    folder_list = ['vtk', 'pytorch', 'vtk_img', 'pytorch_img']
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

    print("RANK: ", rank, len(seg_fns_scatter))
    for seg_fn in seg_fns_scatter:
        img_fn = os.path.join(img_dir, os.path.basename(seg_fn))
        print(img_fn, seg_fn)
        name = os.path.basename(seg_fn).split('.')[0]
        mesh, sdf_v, sdf_v_py = create_from_segmentation(seg_fn, (512, 512, 512), (128, 128, 128), ref_fn, r_ids)
        if mesh is None:
            continue
        write_vtk_image(sdf_v, os.path.join(out_dir, 'vtk', '{}.vti'.format(name)))
        write_vtk_polydata(mesh, os.path.join(out_dir, 'vtk', '{}.vtp'.format(name)))
        pickle.dump(sdf_v_py.astype(np.float32),open(os.path.join(out_dir, 'pytorch', '{}.pkl'.format(name)), 'wb'))

