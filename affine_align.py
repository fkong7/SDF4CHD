import os
import glob
import sys
import argparse
import SimpleITK as sitk
import shutil

def affine_register(fixed_seg, moving_seg):
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_seg)
    p_map_1 = sitk.GetDefaultParameterMap('translation')
    p_map_1['Metric'] = ['AdvancedMeanSquares']
    p_map_2 = sitk.GetDefaultParameterMap('rigid')
    #p_map_2 = sitk.GetDefaultParameterMap('affine')
    p_map_2['Metric'] = ['AdvancedMeanSquares']
    elastixImageFilter.SetParameterMap(p_map_1)
    elastixImageFilter.AddParameterMap(p_map_2)
    elastixImageFilter.SetMovingImage(moving_seg)
    elastixImageFilter.Execute()

    parameter_map = elastixImageFilter.GetTransformParameterMap()
    return parameter_map

def image_transform(parameter_map, moving, order=1):
    if order==0:
        parameter_map[0]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
        parameter_map[1]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    else:
        parameter_map[0]["ResampleInterpolator"] = ["FinalLinearInterpolator"]
        parameter_map[1]["ResampleInterpolator"] = ["FinalLinearInterpolator"]
    return sitk.Transformix(moving, parameter_map)

def write_parameter_map(parameter_map, fn):
    for i, para_map in enumerate(parameter_map):
        para_map_fn = os.path.splitext(fn)[0]+'_%d.txt' % i
        sitk.WriteParameterFile(para_map, para_map_fn)

if __name__ == '__main__':
    ref_id = '/Users/fanweikong/Documents/Modeling/CHD/datasets/imageCHDcleaned/masks_all/ct_1001_image.nii.gz'
    img_dir = '/Users/fanweikong/Documents/Modeling/CHD/datasets/imageCHDcleaned/masks_all/img'
    seg_dir = '/Users/fanweikong/Documents/Modeling/CHD/datasets/imageCHDcleaned/masks_all/seg'
    out_dir = '/Users/fanweikong/Documents/Modeling/CHD/datasets/imageCHDcleaned/masks_all_aligned'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.makedirs(os.path.join(out_dir, 'img'))
        os.makedirs(os.path.join(out_dir, 'seg'))
        os.makedirs(os.path.join(out_dir, 'param'))

    seg_fns = glob.glob(os.path.join(seg_dir, '*.nii.gz'))

    ref_seg = sitk.ReadImage(ref_id) 
    failed_list = []
    for seg_fn in seg_fns:
        im_fn = os.path.join(im_dir, os.path.basename(seg_fn))
        print(seg_fn, im_fn)
        param_fn = glob.glob(os.path.join(out_dir, 'param', os.path.basename(seg_fn).split('.')[0]+'*.txt'))
        if len(param_fn) == 0:
            try:
                moving_seg = sitk.ReadImage(seg_fn)
                moving_img = sitk.ReadImage(im_fn)
                param_map = affine_register(ref_seg, moving_seg)
                aligned_seg = image_transform(param_map, moving_seg, order=0)
                aligned_im = image_transform(param_map, moving_img, order=1)
                sitk.WriteImage(aligned_im, os.path.join(out_dir, 'img', os.path.basename(seg_fn)))
                sitk.WriteImage(aligned_seg, os.path.join(out_dir, 'seg', os.path.basename(seg_fn)))
                write_parameter_map(param_map, os.path.join(out_dir, 'param', os.path.basename(seg_fn)))
            except Exception as e:
                print(e)
                failed_list.append(os.path.basename(seg_fn))
                shutil.copy(seg_fn, os.path.join(out_dir, 'seg', os.path.basename(seg_fn)))
                shutil.copy(im_fn, os.path.join(out_dir, 'img', os.path.basename(seg_fn)))

    with open(os.path.join(out_dir, 'failed_names.txt'), 'w') as f:
        for n in failed_list:
            f.write(f"{n}\n")



