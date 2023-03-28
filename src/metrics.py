import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import SimpleITK as sitk
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../vtk_utils'))
from vtk_utils import *
import csv

def extract_surface(poly):
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(poly)
    connectivity.ColorRegionsOn()
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    poly = connectivity.GetOutput()
    return poly

def surface_distance(p_surf, g_surf):
    dist_fltr = vtk.vtkDistancePolyDataFilter()
    dist_fltr.SetInputData(1, p_surf)
    dist_fltr.SetInputData(0, g_surf)
    dist_fltr.SignedDistanceOff()
    dist_fltr.Update()
    distance = vtk_to_numpy(dist_fltr.GetOutput().GetPointData().GetArray('Distance'))
    return distance, dist_fltr.GetOutput()

def evaluate_poly_distances(poly, gt, NUM):
    # compute assd and hausdorff distances
    assd_list, haus_list, poly_list = [], [], []
    poly =extract_surface(poly)
    for i in range(NUM):
        poly_i = thresholdPolyData(poly, 'Scalars_', (i+1, i+1),'cell')
        if poly_i.GetNumberOfPoints() == 0:
            print("Mesh based methods.")
            poly_i = thresholdPolyData(poly, 'RegionId', (i, i), 'point')
        gt_i = thresholdPolyData(gt, 'Scalars_', (i+1, i+1),'cell')
        pred2gt_dist, pred2gt = surface_distance(gt_i, poly_i)
        gt2pred_dist, gt2pred = surface_distance(poly_i, gt_i)
        assd = (np.mean(pred2gt_dist)+np.mean(gt2pred_dist))/2
        haus = max(np.max(pred2gt_dist), np.max(gt2pred_dist))
        assd_list.append(assd)
        haus_list.append(haus)
        poly_list.append(pred2gt)

    poly_dist = appendPolyData(poly_list)
    # whole heart
    pred2gt_dist, pred2gt = surface_distance(gt, poly)
    gt2pred_dist, gt2pred = surface_distance(poly, gt)

    assd = (np.mean(pred2gt_dist)+np.mean(gt2pred_dist))/2
    haus = max(np.max(pred2gt_dist), np.max(gt2pred_dist))

    assd_list.insert(0, assd)
    haus_list.insert(0, haus)
    return assd_list, haus_list, poly_dist

def dice_score(pred, true):
    pred = pred.astype(np.int)                                   
    true = true.astype(np.int)                                   
    num_class = np.unique(true)
    
    #change to one hot
    dice_out = [None]*len(num_class)
    for i in range(1, len(num_class)):
        pred_c = pred == num_class[i]                            
        true_c = true == num_class[i]                            
        dice_out[i] = np.sum(pred_c*true_c)*2.0 / (np.sum(pred_c) + np.sum(true_c))
    
    mask =( pred > 0 )+ (true > 0)                               
    dice_out[0] = np.sum((pred==true)[mask]) * 2. / (np.sum(pred>0) + np.sum(true>0))
    return dice_out 

def jaccard_score(pred, true):
    pred = pred.astype(np.int)
    true = true.astype(np.int)
    num_class = np.unique(true)

    #change to one hot
    jac_out = [None]*len(num_class)
    for i in range(1, len(num_class)):
        pred_c = pred == num_class[i]
        true_c = true == num_class[i]
        jac_out[i] = np.sum(pred_c*true_c) / (np.sum(pred_c) + np.sum(true_c)-np.sum(pred_c*true_c))

    mask =( pred > 0 )+ (true > 0)
    jac_out[0] = np.sum((pred==true)[mask]) / (np.sum(pred>0) + np.sum(true>0)-np.sum((pred==true)[mask]))
    return jac_out 

def dice_score_from_sdf(pred_sdf, true):
    pred_seg = np.argmax(pred_sdf, axis=-1) + 1
    pred_seg[np.all(pred_sdf<0.5, axis=-1)] = 0
    return dice_score(pred_seg.transpose(2, 1, 0), true)

def jaccard_score_from_sdf(pred_sdf, true):
    pred_seg = np.argmax(pred_sdf, axis=-1) + 1
    pred_seg[np.all(pred_sdf<0.5, axis=-1)] = 0
    return jaccard_score(pred_seg.transpose(2, 1, 0), true)

