import numpy as np
import torch
from stylegan2_pytorch import ModelLoader
from tqdm import tqdm
import nibabel as nib
import os
import torchio as tio


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


def generate_and_save_images(noise_or_style, input_noise, save_directory, file_name, mode, truncation_psi, affine_matrix_template, img_id=0, version_id=0):
    if mode == "latent":
        styles = loader.noise_to_styles(noise_or_style, trunc_psi=truncation_psi)  # pass through mapping network
    else:
        styles = noise_or_style

    images = loader.styles_to_images_ext_noise(styles, input_noise)  # call the generator on intermediate style vectors

    images = images.clamp_(0., 1.)

    affine_matrix_generated = affine_matrix_template

    image_path = os.path.join(save_directory, f"{file_name}_{img_id}_{version_id}.nii.gz")

    latents_and_noises_path = os.path.join(save_directory, "style_and_noise", f"{file_name}_{img_id}_{version_id}.nii.gz")

    if dimension == "2D":
        img_array = images.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    if dimension == "3D":
        img_array = images.squeeze(0).permute(2, 3, 1, 0).cpu().detach().numpy()

    # clip values between 0 and 1
    img_array = np.clip(img_array, 0, 1)
    nifti_axial_slice = nib.Nifti1Image(img_array, affine_matrix_generated)

    nib.save(nifti_axial_slice, image_path)

    np.save(latents_and_noises_path.replace(".nii.gz", "_style.npy"), styles.cpu().numpy())
    np.save(latents_and_noises_path.replace(".nii.gz", "_noise.npy"), input_noise.cpu().numpy())


# Set the environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda')

# GENERAL CONFIG
mode = "normal"  # select only one from: ["stochastic_variation", "interpolation", "normal"]
dimension = "3D"

test_upsampling_methods = False

if mode == "normal" and test_upsampling_methods:
    upsample_methods = ["SRResnet", "bspline", "no_upsampling"]


top_dir = '/home/user/synthetic_3D_data_TOF_MRA_CoW'  # should include "models" directory

model_folder_name = "Medium_v2"  # name of the folder that contains the models from the selected epoch

model_from_epoch = 23  # model step to use for generation: Example: model_23.pt should be in the models folder in the top_dir

# Example: full path of the model /home/user/synthetic_3D_data_TOF_MRA_CoW/models/Medium_v2/model_23.pt

fmap_max = 1024  # set maximum number of filters set during model training.
                 # For Medium_v2 and Large 1024, For Medium_v1 512, for Small 256.

network_capacity = 8  # set the model capacity parameter, indirectly defines initial number of Conv filters of Generator and Discriminator.
                      # For the configurations Medium_v2, Medium_v1 and Small from the publication this should be set to 8. For Large 16.
                      # Check the stylegan2_pytorch file to see how this parameter defines inital conv fiter number in teh generator and discriminator

num_generate = 1  # Depending on teh mode variable:
                  # Normal: number of images to generate
                  # interpolation: random noise latent vector pairs to perform the interpolation on
                  # stochastic_variation: number of images to manipulate with stochastic variation


truncation_psi = 0.9  # truncation to control image quality/diversity tradeoff

save_directory = os.path.join(top_dir, mode)
# provide template so that the generated volumes have the same affine as the template.
template_path = "./sample_data/TOF_template_NITRC.nii.gz"
affine_matrix_generated = nib.load(template_path).affine

assert mode == "interpolation" or mode == "stochastic_variation" or mode == "normal"

# INTERPOLATION CONFIGURATION
if mode == "interpolation":
    interpolate_styles = False  # if True next to the latent interpolation the interpolations after the mapping network are saved as an alternative
    steps = 100  # number of interpolation steps between two random images

# STOCHASTIC VARIATION CONFIGURATION
if mode == "stochastic_variation":
    num_stochastic_variation = 100
    load_from_style = False  # load an existing style vector to modify the image with stochastic variations
    if load_from_style:
        style_npy_path = "/home/user/style_and_noise/stochastic_variation_0_0_style.npy"
        num_generate = 1

# USUAL GENERATION
loader = ModelLoader(
    base_dir=top_dir,   # path to where the command line tool is invoked, contains folders "models", "results"
    name=model_folder_name,  # the project name/ model folder name
    fmap_max=fmap_max,
    network_capacity=network_capacity,
    load_from=model_from_epoch)

os.makedirs(save_directory, exist_ok=True)
os.makedirs(os.path.join(save_directory, "style_and_noise"), exist_ok=True)

# Load Super-Resolution-ResNet
device = torch.device('cuda')

for i in tqdm(range(num_generate), desc="Generating Images"):
    if mode == "interpolation":
        if dimension == "2D":
            input_noise = torch.FloatTensor(1, loader.model.image_size, loader.model.image_size, 1).uniform_(0., 1.).to(device)
        if dimension == "3D":
            input_noise = torch.FloatTensor(1, loader.model.image_size, loader.model.image_size,
                                            loader.model.image_size, 1).uniform_(0., 1.).to(device)
        noise_low = torch.randn(1, 512).to(device)
        noise_high = torch.randn(1, 512).to(device)


        styles_low = loader.noise_to_styles(noise_low, trunc_psi=truncation_psi)  # pass through mapping network
        styles_high = loader.noise_to_styles(noise_high, trunc_psi=truncation_psi)  # pass through mapping network

        ratios = torch.linspace(0., 1., steps)

        for v, ratio in enumerate(ratios):
            interp_latents = slerp(ratio, noise_low, noise_high)

            generate_and_save_images(interp_latents, input_noise, save_directory, "noise_interpolation", "latent", truncation_psi, affine_matrix_generated, i, v)

            if interpolate_styles:
                interp_styles = slerp(ratio, styles_low, styles_high)
                generate_and_save_images(interp_styles, input_noise, save_directory, "style_interpolation", "style",  truncation_psi, affine_matrix_generated, i, v)

    if mode == "stochastic_variation":
        if load_from_style:
            styles = torch.from_numpy(np.load(style_npy_path)).to(device)
        else:
            noise = torch.randn(1, 512).to(device)
            styles = loader.noise_to_styles(noise, trunc_psi=truncation_psi)

        for v in range(num_stochastic_variation):
            if dimension == "2D":
                input_noise = torch.FloatTensor(1, loader.model.image_size, loader.model.image_size,  1).uniform_(0., 1.).to(device)
            if dimension == "3D":
                input_noise = torch.FloatTensor(1, loader.model.image_size, loader.model.image_size, loader.model.image_size, 1).uniform_(0., 1.).to(device)
            generate_and_save_images(styles, input_noise, save_directory, "stochastic_variation", "style",  truncation_psi, affine_matrix_generated, i, v)

    if mode == "normal":
        v = 0  # there are no versions/manipulations of the  generated image
        noise = torch.randn(1, 512).to(device)

        styles = loader.noise_to_styles(noise, trunc_psi=truncation_psi)

        if dimension == "2D":
            input_noise = torch.FloatTensor(1, loader.model.image_size, loader.model.image_size,  1).uniform_(0., 1.).to(device)
        if dimension == "3D":
            input_noise = torch.FloatTensor(1, loader.model.image_size, loader.model.image_size, loader.model.image_size, 1).uniform_(0., 1.).to(device)

        generate_and_save_images(styles, input_noise, save_directory, "normal", "style", truncation_psi, affine_matrix_generated, i, v)


