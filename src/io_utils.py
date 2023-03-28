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
from sdf import vtk_marching_cube 
import torch
import shutil
import csv
import SimpleITK as sitk

def save_ckp(model, optimizers, epoch, is_best, checkpoint_dir, best_model_dir):
    checkpoint = {
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    }
    for i, opt in enumerate(optimizers):
        checkpoint['optimizer_{}'.format(i)] = opt.state_dict()
    f_path = os.path.join(checkpoint_dir, 'net_{}.pt'.format(checkpoint['epoch']))
    torch.save(checkpoint, f_path)
    if is_best:
        best_fpath = os.path.join(best_model_dir, 'best_model.pt')
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizers):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    for i, opt in enumerate(optimizers):
        optimizers[i].load_state_dict(checkpoint['optimizer_{}'.format(i)])
    return model, optimizers, checkpoint['epoch']

def write_sampled_point(points, point_values, fn, flip=True):
    pts_py = (np.squeeze(points.detach().cpu().numpy()) + 1.)/2.
    if flip:
        pts_py = np.flip(pts_py, -1)
    #print(pts_py, v_py)
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(pts_py))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)

    if point_values is not None:
        v_py = np.squeeze(point_values.detach().cpu().numpy())
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


def write_sdf_to_vtk_mesh(sdf, fn, thresh, decimate=0.):
    mesh_list = []
    region_ids = []
    for i in range(sdf.shape[-1]):
        img = vtk.vtkImageData()
        img.SetDimensions(sdf.shape[:-1])
        img.SetSpacing(np.ones(3)/np.array(sdf.shape[:-1]))
        img.SetOrigin(np.zeros(3))
        img.GetPointData().SetScalars(numpy_to_vtk(sdf[:, :, :, i].transpose(2, 1, 0).flatten()))
        mesh = vtk_marching_cube(img, thresh)
        # tmp flip
        mesh_coords = np.flip(vtk_to_numpy(mesh.GetPoints().GetData()), axis=-1)
        #mesh_coords[:, 0] *= -1
        #mesh_coords[:, 2] *= -1
        mesh.GetPoints().SetData(numpy_to_vtk(mesh_coords))
        region_ids += [i]*mesh.GetNumberOfPoints()
        mesh_list.append(mesh)
    mesh_all = vtk_utils.appendPolyData(mesh_list)
    region_id_arr = numpy_to_vtk(np.array(region_ids))
    region_id_arr.SetName('RegionId')
    mesh_all.GetPointData().AddArray(region_id_arr)
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
