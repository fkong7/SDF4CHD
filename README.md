# SDF4CHD

## Step 1: Rigid registration to align shapes
The following script applies the rigid registration function from SimpleElastix to align the shapes before training. SimpleElastix is now available using pip <https://pypi.org/project/SimpleITK-SimpleElastix/>. 
```
python data/affine_align.py
```
## Step 2: Apply augmentation to the shapes
Since our dataset size is very small, we apply augmentation to the shapes prior to training. We found that doing so improved convergence. 
We used scripts from [MeshDeformNet](https://github.com/fkong7/MeshDeformNet/blob/main/data/data_augmentation.py) to do the augmentation. Here are the parameters we used for this project.
```
    params_affine = { 
            'scale_range': [1., 1.2],
            'rot_range': [-5., 5.],
            'trans_range': [-0., 0.],
            'shear_range': [-0.1, 0.1],
            'flip_prob': 0.
            }   
    params_bspline = { 
            'num_ctrl_pts': 16, 
            'stdev': 3
            }  
 ```
 We augmented 20 copies per sample. The elastic augmentation is slow and thus parallelization is recommended. Example usage is below. 
 ```
 mpirun -n 24 python  /home/users/fwkong/MeshDeformNet/data/data_augmentation.py \
    --im_dir /scratch/users/fwkong/CHD/imageCHDCleanedOriginal_aligned/masks_all_aligned/img \
    --seg_dir /scratch/users/fwkong/CHD/imageCHDCleanedOriginal_aligned/masks_all_aligned/seg\
    --out_dir /scratch/users/fwkong/CHD/imageCHDCleanedOriginal_aligned_Aug/ \
    --modality ct \
    --mode train \
    --num 20
 ```
 
 ## Step 3: Preprocess data and convert segmentation to SDFs
 
 ```
 python create_sdfdataset.py
 ```
 
 ## Step 4: Training
 We used the auto-decoder structure proposed by [DeepSDF](https://github.com/facebookresearch/DeepSDF). During training we create a optimizable latent embedding for all training samples (including the augmented ones) and during testing we extract the embeddings that correspond to the original samples for inference. The config file for training our network is here `config/auto_decoder.yml`. However, it's also possible to use a auto-encoder structure or have a image encoder to predict shapes from image data. Here is the config file to the auto-encoder network: `config/auto_encoder.yml`. We applied the encoder on image data during training for a segmentation project, while it is also possible to use the SDFs as inputs by setting the `encoder_in_dim` to be the same as the `out_dim`. 
 ```
 python train_gen.py --config config/auto_decoder.yml
 ```
