import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../vtk_utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from vtk_utils import vtk_utils
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
import matplotlib.pyplot as plt
import pickle
import torch
import shutil
import csv
import SimpleITK as sitk
from pre_process import centering, resample_spacing
def save_ckp_single(model, optimizer, scheduler, epoch, checkpoint_dir):
    checkpoint = {
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict()
    }
    if isinstance(model, torch.nn.DataParallel):
        checkpoint['state_dict'] = model.module.state_dict()
    f_path = os.path.join(checkpoint_dir, 'net_{}.pt'.format(checkpoint['epoch']))
    torch.save(checkpoint, f_path)
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'net.pt'))

def load_ckp_single(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch']

def save_ckp(model, optimizers, schedulers, epoch, checkpoint_dir, name='net'):
    checkpoint = {
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'optimizers': [o.state_dict() for o in optimizers],
    'schedulers': [s.state_dict() for s in schedulers]
    }
    if isinstance(model, torch.nn.DataParallel):
        checkpoint['state_dict'] = model.module.state_dict()
    f_path = os.path.join(checkpoint_dir, '{}_{}.pt'.format(name, checkpoint['epoch']))
    torch.save(checkpoint, f_path)
    torch.save(checkpoint, os.path.join(checkpoint_dir, '{}.pt'.format(name)))

def load_ckp(checkpoint_fpath, model, optimizers, schedulers):
    checkpoint = torch.load(checkpoint_fpath)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    for i, opt in enumerate(optimizers):
        optimizers[i].load_state_dict(checkpoint['optimizers'][i])
    for i, sch in enumerate(schedulers):
        schedulers[i].load_state_dict(checkpoint['schedulers'][i])
    return model, optimizers, schedulers, checkpoint['epoch']

def write_sampled_point(points, point_values, fn, flip=True):
    pts_py = (np.squeeze(points.detach().cpu().numpy()) + 1.)/2.
    if flip:
        pts_py = np.flip(pts_py, -1)
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(pts_py))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)

    if point_values is not None:
        v_py = np.squeeze(point_values.detach().cpu().numpy()).transpose()
        v_arr = numpy_to_vtk(v_py)
        v_arr.SetName('occupancy')
        poly.GetPointData().AddArray(v_arr)
    verts = vtk.vtkVertexGlyphFilter()
    verts.AddInputData(poly)
    verts.Update()
    poly_v = verts.GetOutput()

    vtk_utils.write_vtk_polydata(poly_v, fn)


def write_ply_triangle(fn, vertices, triangles):
    if len(vertices)>0:
        mesh = vtk.vtkPolyData()
        conn = numpy_to_vtk(triangles.astype(np.int64))
        ids = (np.ones((triangles.shape[0],1))*3).astype(np.int64)
        conn = np.concatenate((ids, conn), axis=-1)
        vtk_arr = numpy_to_vtkIdTypeArray(conn)

        c_arr = vtk.vtkCellArray()
        c_arr.SetCells(triangles.shape[0], vtk_arr)

        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(vertices))

        mesh.SetPoints(pts)
        mesh.SetPolys(c_arr)

        w = vtk.vtkXMLPolyDataWriter()
        w.SetInputData(mesh)
        w.SetFileName(fn)
        w.Update()
        w.Write()

def sdf_to_vtk_image(sdf):
    img = vtk.vtkImageData()
    img.SetDimensions(sdf.shape[:-1])
    img.SetSpacing(np.ones(3)/np.array(sdf.shape[:-1]))
    img.SetOrigin(np.zeros(3))
    sdf_r = sdf.reshape((np.prod(sdf.shape[:-1]), sdf.shape[-1]))
    for i in range(sdf.shape[-1]):
        sdf_vtk = numpy_to_vtk(sdf_r[:, i])
        sdf_vtk.SetName('id{}'.format(i))
        img.GetPointData().AddArray(sdf_vtk)
    return img

def vtk_image_to_sdf(vtk_image):
    sdf_list = []
    num_class = vtk_image.GetPointData().GetNumberOfArrays()
    for i in range(num_class):
        name = 'id{}'.format(i)
        sdf_i = vtk_to_numpy(vtk_image.GetPointData().GetArray(name))
        sdf_list.append(sdf_i)
    x, y, z = vtk_image.GetDimensions()
    sdf = np.array(sdf_list).transpose(1, 0).reshape(x, y, z, num_class)
    return sdf

def write_vtk_image(sdf, fn, info=None):
    img = sdf_to_vtk_image(sdf)
    if info is not None:
        img.SetSpacing(info['spacing'])
        img.SetOrigin(info['origin'])
    w = vtk.vtkXMLImageDataWriter()
    w.SetInputData(img)
    w.SetFileName(fn)
    w.Update()
    w.Write()

def write_nifty_image(sdf, fn):
    sdf = np.squeeze(sdf)
    img = sitk.GetImageFromArray(sdf.astype(np.float32))
    sitk.WriteImage(img, fn)


def write_sdf_to_vtk_mesh(sdf, fn, thresh, decimate=0., keep_largest=False):
    mesh_list = []
    region_ids = []
    for i in range(sdf.shape[-1]):
        img = vtk.vtkImageData()
        img.SetDimensions(sdf.shape[:-1])
        img.SetSpacing(np.ones(3)/np.array(sdf.shape[:-1]))
        img.SetOrigin(np.zeros(3))
        img.GetPointData().SetScalars(numpy_to_vtk(sdf[:, :, :, i].transpose(2, 1, 0).flatten()))
        if keep_largest:
            img = vtk_utils.image_largest_connected(img)
            mesh = vtk_utils.vtk_marching_cube(img, 0, 1)
        else:
            mesh = vtk_utils.vtk_marching_cube_continuous(img, thresh)
        # tmp flip
        mesh_coords = vtk_to_numpy(mesh.GetPoints().GetData())
        #mesh_coords[:, 0] *= -1
        #mesh_coords[:, 2] *= -1
        mesh.GetPoints().SetData(numpy_to_vtk(mesh_coords))
        region_ids += [i]*mesh.GetNumberOfPoints()
        mesh_list.append(mesh)
    mesh_all = vtk_utils.appendPolyData(mesh_list)
    region_id_arr = numpy_to_vtk(np.array(region_ids))
    region_id_arr.SetName('RegionId')
    mesh_all.GetPointData().AddArray(region_id_arr)
    if keep_largest:
        mesh_all = vtk_utils.smooth_polydata(mesh_all, smoothingFactor=0.5)
    if decimate > 0.:
        mesh_all = vtk_utils.decimation(mesh_all, decimate)
    vtk_utils.write_vtk_polydata(mesh_all, fn)
    return mesh_all

def write_seg_to_vtk_mesh(seg, fn):
    mesh_list = []
    ids = np.unique(seg)
    
    img = vtk.vtkImageData()
    img.SetDimensions(seg.shape)
    img.SetSpacing(np.ones(3)/np.array(seg.shape))
    img.SetOrigin(np.zeros(3))
    img.GetPointData().SetScalars(numpy_to_vtk(seg.transpose(2, 1, 0).flatten()))
  
    region_ids = np.array(())
    for i in ids:
        if i==0:
            continue
        mesh = vtk_utils.vtk_marching_cube(img,0, i)
        # tmp flip
        mesh_coords = vtk_to_numpy(mesh.GetPoints().GetData())
        mesh.GetPoints().SetData(numpy_to_vtk(mesh_coords))
        region_ids = np.append(region_ids, np.ones(mesh.GetNumberOfPoints()))
        mesh_list.append(mesh)
        print("Region id: ", i, mesh.GetNumberOfPoints())
    mesh_all = vtk_utils.appendPolyData(mesh_list)
    region_id_arr = numpy_to_vtk(region_ids.astype(int))
    region_id_arr.SetName('RegionId')
    mesh_all.GetPointData().AddArray(region_id_arr)
    vtk_utils.write_vtk_polydata(mesh_all, fn)
    return mesh_all

def write_sdf_to_seg(sdf, fn, thresh):
    seg = np.argmax(sdf, axis=-1) + 1
    seg[np.all(sdf<thresh, axis=-1)] = 0
    seg_im = sitk.GetImageFromArray(seg.astype(np.uint8))
    sitk.WriteImage(seg_im, fn)

def plot_loss_curves(loss_dict, title, out_fn):
    for key in loss_dict.keys():
        plt.figure(figsize=(10,5))
        plt.title(title)
        plt.plot(loss_dict[key], label=key)
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(out_fn[:-4]+key+'.png')
    pickle.dump(loss_dict, open(os.path.splitext(out_fn)[0] + '_history', 'wb'))

def write_scores(csv_path,scores, header=('Dice', 'ASSD')): 
    with open(csv_path, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(header)
        for i in range(len(scores)):
            writer.writerow(tuple(scores[i]))
            print(scores[i])
    writeFile.close()
def import_nifty_img(img_fn, size=(128, 128, 128), order=0):
    img = sitk.ReadImage(img_fn)
    img_r = resample_spacing(img, template_size=size, order=order)[0]
    img_ds_vtk = vtk_utils.exportSitk2VTK(img_r, img_r.GetSpacing())[0]
    x, y, z = img_ds_vtk.GetDimensions()
    py_img = vtk_to_numpy(img_ds_vtk.GetPointData().GetScalars()).reshape(z, y, x).transpose(2, 1, 0).astype(np.float32)
    return py_img

def write_numpy_image_to_nifty(py_img, ref_img_fn, order=0):
    ref_img = sitk.ReadImage(ref_img_fn)
    img = sitk.GetImageFromArray(py_img.transpose(2, 1, 0))
    ref_img_ds = resample_spacing(ref_img, template_size=img.GetSize(), order=0)[0]
    img.CopyInformation(ref_img_ds)
    img = centering(img, ref_img, order=order)
    return img

