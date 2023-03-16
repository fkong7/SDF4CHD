import numpy as np
import SimpleITK as sitk
from copy import deepcopy
def resample(sitkIm, resolution = (0.5, 0.5, 0.5),order=1,dim=3):
    if type(sitkIm) is str:
        image = sitk.ReadImage(sitkIm)
    else:
        image = sitkIm
    resample = sitk.ResampleImageFilter()
    if order==1:
        resample.SetInterpolator(sitk.sitkLinear)
    else:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputSpacing(resolution)

    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = np.array(image.GetSpacing())
    new_size = orig_size*(orig_spacing/np.array(resolution))
    new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    newimage = resample.Execute(image)
    
    return newimage


def transform_func(image, reference_image, transform, order=1):
    # Output vtk_image Origin, Spacing, Size, Direction are taken from the reference
    # vtk_image in this call to Resample
    if order ==1:
        interpolator = sitk.sitkLinear
    elif order == 0:
        interpolator = sitk.sitkNearestNeighbor
    elif order ==3:
        interpolator = sitk.sitkBSpline
    default_value = 0
    try:
        resampled = sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)
    except Exception as e: print(e)

    return resampled

def reference_image_build(spacing, size, template_size, dim):
    #template size: vtk_image(array) dimension to resize to: a list of three elements
    reference_size = template_size
    reference_spacing = np.array(size)/np.array(template_size)*np.array(spacing)
    reference_spacing = np.mean(reference_spacing)*np.ones(3)
    #reference_size = size
    reference_image = sitk.Image(reference_size, 0)
    reference_image.SetOrigin(np.zeros(3))
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(np.eye(3).ravel())
    return reference_image

def centering(img, ref_img, order=1):
    dimension = img.GetDimension()
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - ref_img.GetOrigin())
    # Modify the transformation to align the centers of the original and reference vtk_image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    reference_center = np.array(ref_img.TransformContinuousIndexToPhysicalPoint(np.array(ref_img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform = sitk.CompositeTransform([centered_transform, centering_transform])

    return transform_func(img, ref_img, centered_transform, order)

def isometric_transform(image, ref_img, orig_direction, order=1, target=None):
    # transform vtk_image volume to orientation of eye(dim)
    dim = ref_img.GetDimension()
    affine = sitk.AffineTransform(dim)
    if target is None:
        target = np.eye(dim)
    
    ori = np.reshape(orig_direction, np.eye(dim).shape)
    target = np.reshape(target, np.eye(dim).shape)
    affine.SetMatrix(np.matmul(target,np.linalg.inv(ori)).ravel())
    affine.SetCenter(ref_img.TransformContinuousIndexToPhysicalPoint(np.array(ref_img.GetSize())/2.0))
    #affine.SetMatrix(vtk_image.GetDirection())
    return transform_func(image, ref_img, affine, order)

def resample_spacing(sitkIm, resolution=0.5, dim=3, template_size=(256, 256, 256), order=1):
    if type(sitkIm) is str:
        image = sitk.ReadImage(sitkIm)
    else:
        image = sitkIm
    orig_direction = image.GetDirection()
    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = np.array(image.GetSpacing())
    new_size = orig_size*(orig_spacing/np.array(resolution))
    new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    new_size = np.abs(np.matmul(np.reshape(orig_direction, (3,3)), np.array(new_size)))
    ref_img = reference_image_build(resolution, new_size, template_size, dim)
    centered = centering(image, ref_img, order)
    transformed = isometric_transform(centered, ref_img, orig_direction, order)
    return transformed, ref_img

def rescale_intensity(slice_im,m,limit):
    if type(slice_im) != np.ndarray:
        raise RuntimeError("Input vtk_image is not numpy array")
    #slice_im: numpy array
    #m: modality, ct or mr
    if m =="ct":
        rng = abs(limit[0]-limit[1])
        threshold = rng/2
        slice_im[slice_im>limit[0]] = limit[0]
        slice_im[slice_im<limit[1]] = limit[1]
        #(slice_im-threshold-np.min(slice_im))/threshold
        slice_im = slice_im/threshold
    elif m=="mr":
        #slice_im[slice_im>limit[0]*2] = limit[0]*2
        #rng = np.max(slice_im) - np.min(slice_im)
        pls = np.unique(slice_im)
        #upper = np.percentile(pls, 99)
        #lower = np.percentile(pls, 10)
        upper = np.percentile(slice_im, 99)
        lower = np.percentile(slice_im, 20)
        slice_im[slice_im>upper] = upper
        slice_im[slice_im<lower] = lower
        slice_im -= int(lower)
        rng = upper - lower
        slice_im = slice_im/rng*2
        slice_im -= 1
    return slice_im
