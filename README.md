# Generative Modelling of the Circle of Willis using 3D StyleGAN

The proposed 3D modification of the StyleGANv2 (original paper: https://arxiv.org/abs/1912.04958 and implementation by https://github.com/lucidrains/stylegan2-pytorch) allows the generation of single-channel 3D medical imaging data. 

Paper and citation: 
Aydin, Orhun Utku, Adam Hilbert, Alexander Koch, Felix Lohrke, Jana Rieger, Satoru Tanioka, and Dietmar Frey. “Generative Modeling of the Circle of Willis Using 3D-StyleGAN.” NeuroImage, November 23, 2024, 120936. https://doi.org/10.1016/j.neuroimage.2024.120936.

For a detailed overview of our model's specifications and performance, please see our [model card](model-card.md).

## Generated 3D TOF MRA Example
The generated 3D TOF MRA volume of the Circle of Willis has a resolution of 128x128x32 and is anatomically realistic with respect to vessel neuroanatomy.
<img src="sample_data/example_3D.gif" width="800" height="800" alt="Generated TOF MRA example">

### Style space interpolation
The trained 3D StylGAN was used to generate 100 TOF MRA volumes by interpolating in 100 steps between two latent vectors using spherical interpolation.  
The middle axial slices of the 100 TOF MRA volumes are shown in the gif below. The smooth interpolation is indicative of model generalization and little overfitting [Zhao et al.](http://arxiv.org/abs/2006.10738)  

<img src="sample_data/interpolation.gif" width="800" height="800" alt="Interpolation between two latent vectors">

### Stochastic variation 
The following GIF shows middle axial slices of 100 generated TOF MRA volumes where the style vector is constant but the noise input of the generator is varied.  
The vessel anatomy remains mostly unchanged, whereas background details show significant variation.

<img src="sample_data/stochastic_variation.gif" width="800" height="800" alt="Stochastic variation example">


## Installation

Use the [environment.yml](environment.yml) file to install necessary packages, or install them manually. 

```python
conda env create -f environment.yml
```

Tested with CUDA version 12.2. A GPU with 24 GB VRAM is recommended, GPUs with lower VRAM can be used with gradient accumulation.

## Preprocessing
The model requires a constant input volume size for all real training volumes. In our paper we defined a region of interest (ROI) of (32,128,128). The [preprocessing script](preprocessing/preprocessing.py) for the TOF MRA volumes performs skullstripping, registration to a custom TOF MRA template, and cropping to the required ROI size. While the preprocessing step is optional, we hypothesize that StyleGAN performs better on centered/registered datasets (such as [FFHQ](https://github.com/NVlabs/ffhq-dataset) or registered medical data).  

## Usage
### Training
The [training script](stylegan2_pytorch.py) can be used with command line arguments. All real training data should be saved in NIfTI file format (nii.gz) in a single folder.

```python
python stylegan2_pytorch.py --data folder_containing_3D_nifti_128_128_32 \
                            --name PROJECT_NAME \
                            --results_dir folder_to_save_results \
                            --models_dir folder_to_save_models \
                            --image-size 128 \
                            --network_capacity 8 \
                            --fmap_max 1024 \
                            --gradient-accumulate-every 4 \
                            --batch-size 8 \
                            --learning_rate 4e-05 \
                            --aug-prob 0.5 \
                            --aug-types [translation,cutout] \
                            --dataset_aug_prob 0.5 \
                            --save_every 3000 \
                            --calculate_fid_every 3000 \
                            --calculate_fid_num_images 1782 \
                            --trunc_psi 1 \
                            --ttur_mult 1.5 \
                            --fp16 True 
```

### Generation 
After training a 3D StyleGAN model,the [generation script](CoW_generator.py) can be used to generate 3D TOF MRA volumes. You can set the "mode" variable in the file to one of the following ["normal", "stochastic_variation", "interpolation"].  
The variable "num_generate" defines how the number of TOF MRA volumes to be generated.  


### Evaluation 
The [evaluation script](MedicalNet/evaluation.py) calculates Frechet Inception Distance (FID), MedicalNet Distance (MD) and Area Under the Curve of the Precision and Recall Curve for Distributions (AUC-PRD) given two folders containing 3D niftis (generated and real).


## Additional features
Features like contrastive loss regularization, attention layers for the discriminator, feature quantization, top-k training etc. have not been tested. Details can be found in the https://github.com/lucidrains/stylegan2-pytorch and respective linked publications.

## References
Generative Modeling of the Circle of Willis Using 3D-StyleGAN
Orhun Utku Aydin, Adam Hilbert, Alexander Koch, Felix Lohrke, Jana Rieger, Satoru Tanioka, Dietmar Frey
medRxiv 2024.04.02.24305197; doi: https://doi.org/10.1101/2024.04.02.24305197 

## License
see LICENSE.txt file and:
[MIT](https://choosealicense.com/licenses/mit/)