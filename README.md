# SDF4CHD

Here is the source code of our conference paper ''Type and Shape Disentangled Generative Modeling for Congenital Heart Defects'', which is accepted to The Statistical Atlases and Computational Modeling of the Heart (STACOM) workshop. 

The source code of the extended version of this paper, ''[SDF4CHD: Generative Modeling of Cardiac Anatomies with Congenital Heart Defects](https://arxiv.org/abs/2311.00332)'' will be released soon at this repository. 

## Step 1: Rigid registration to align shapes
The following script applies the rigid registration function from SimpleElastix to align the shapes before training. SimpleElastix is now available using pip <https://pypi.org/project/SimpleITK-SimpleElastix/>. 
```
python data/affine_align.py
```
## Step 2: Preprocess data and convert segmentation to SDFs
 
 ```
 python create_sdfdataset.py
 ```
 
 ## Step 3: Training
 We used the auto-decoder structure proposed by [DeepSDF](https://github.com/facebookresearch/DeepSDF). During training we create a optimizable latent embedding for all training samples (including the augmented ones) and during testing we extract the embeddings that correspond to the original samples for inference. The config file for training our network is here `config/auto_decoder.yml`. However, it's also possible to use a auto-encoder structure or have a image encoder to predict shapes from image data. Here is the config file to the auto-encoder network: `config/auto_encoder.yml`. We applied the encoder on image data during training for a segmentation project, while it is also possible to use the SDFs as inputs by setting the `encoder_in_dim` to be the same as the `out_dim`. 
 ```
 python train_gen.py --config config/auto_decoder.yml
 ```
