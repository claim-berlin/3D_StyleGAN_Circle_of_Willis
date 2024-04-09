from functools import partial
import random
import torch
import torch.nn.functional as F

# Differentiable augmentations were adapted from https://github.com/lucidrains/stylegan2-pytorch and
# https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_pytorch.py

def DiffAugment(x, types=[]):
    for p in types:
        for f in AUGMENT_FNS[p]:
            x = f(x)
    return x.contiguous()

def rand_brightness(x, scale):
    x = x + (torch.rand(x.size(0), 1, 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5) * scale
    return x

def rand_contrast(x, scale):
    x_mean = x.mean(dim=[1, 2, 3, 4], keepdim=True)
    x = (x - x_mean) * (((torch.rand(x.size(0), 1, 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5) * 2.0 * scale) + 1.0) + x_mean
    return x

def rand_translation(x, ratio=0.125):
    shift_x, shift_y, shift_z = int(x.size(3) * ratio + 0.5), int(x.size(4) * ratio + 0.5), int(x.size(2) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1, 1], device=x.device)
    translation_z = torch.randint(-shift_z, shift_z + 1, size=[x.size(0), 1, 1, 1], device=x.device)

    grid_batch, grid_z, grid_y, grid_x = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        torch.arange(x.size(4), dtype=torch.long, device=x.device),
    )

    grid_x = torch.clamp(grid_x + translation_x, 0, x.size(4) - 1)
    grid_y = torch.clamp(grid_y + translation_y, 0, x.size(3) - 1)
    grid_z = torch.clamp(grid_z + translation_z, 0, x.size(2) - 1)


    x_pad = F.pad(x, [1, 1, 1, 1, 1, 1, 0, 0])
    x = x_pad.permute(0, 2, 3, 4, 1).contiguous()[grid_batch, grid_z, grid_y, grid_x].permute(0, 4, 1, 2, 3)
    return x

def rand_offset(x, ratio=1, ratio_h=1, ratio_v=1, ratio_d=1):
    d, w, h = x.size(2), x.size(3), x.size(4)

    imgs = []
    for img in x.unbind(dim = 0):
        max_h = int(w * ratio * ratio_h)
        max_v = int(h * ratio * ratio_v)
        max_d = int(d * ratio * ratio_d)

        value_h = random.randint(0, max_h) * 2 - max_h
        value_v = random.randint(0, max_v) * 2 - max_v
        value_d = random.randint(0, max_d) * 2 - max_d

        if abs(value_h) > 0:
            img = torch.roll(img, value_h, 3)

        if abs(value_v) > 0:
            img = torch.roll(img, value_v, 4)

        if abs(value_d) > 0:
            img = torch.roll(img, value_d, 2)

        imgs.append(img)

    return torch.stack(imgs)

def rand_offset_h(x, ratio=1):
    return rand_offset(x, ratio=1, ratio_h=ratio, ratio_v=0, ratio_d=0)

def rand_offset_v(x, ratio=1):
    return rand_offset(x, ratio=1, ratio_h=0, ratio_v=ratio, ratio_d=0)

def rand_offset_d(x, ratio=1):
    return rand_offset(x, ratio=1, ratio_h=0, ratio_v=0, ratio_d=ratio)


def rand_cutout(x, ratio=0.5):
    cutout_size = (int(x.size(2) * ratio + 0.5),
                   int(x.size(3) * ratio + 0.5),
                   int(x.size(4) * ratio + 0.5))


    offset_x = torch.randint(0, x.size(4) + (1 - cutout_size[2] % 2), size=[x.size(0), 1, 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1, 1], device=x.device)
    offset_z = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1, 1], device=x.device)

    grid_batch, grid_z, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[2], dtype=torch.long, device=x.device),
    )

    grid_x = torch.clamp(grid_x + offset_x - cutout_size[2] // 2, min=0, max=x.size(4) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    grid_z = torch.clamp(grid_z + offset_z - cutout_size[0] // 2, min=0, max=x.size(2) - 1)

    mask = torch.ones(x.size(0), x.size(2), x.size(3), x.size(4), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_z, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

AUGMENT_FNS = {
    'contrast': [partial(rand_brightness, scale=1.), partial(rand_contrast, scale=0.5)],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}

# TEST AUGMENTATIONS
import nibabel as nib
import torch
import torch.nn.functional as F
from functools import partial
import os
import random


def load_nifti_file(file_path):
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    tensor_data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # Add channel and batch dimension
    return tensor_data, nifti_img.affine

def save_nifti_file(tensor_data, affine, output_path):
    data = tensor_data.squeeze().cpu().numpy()  # Remove channel and batch dimension
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)

def apply_augmentations(input_tensor, augmentations, affine):
    augmented_tensors = {}
    for aug_name, funcs in augmentations.items():
        aug_tensor = input_tensor.clone()
        for func in funcs:
            aug_tensor = func(aug_tensor)
        augmented_tensors[aug_name] = aug_tensor
    return augmented_tensors

def main(input_file_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_tensor, affine = load_nifti_file(input_file_path)

    augmented_tensors = apply_augmentations(input_tensor, AUGMENT_FNS, affine)

    for aug_name, aug_tensor in augmented_tensors.items():
        output_path = os.path.join(output_dir, f"{aug_name}_augmented.nii")
        save_nifti_file(aug_tensor, affine, output_path)
        print(f"Saved {aug_name} augmentation at {output_path}")

if __name__ == "__main__":
    input_file_path = "/home/user/test_nifti_file.nii.gz"
    output_dir = "/home/user/augmentation_tested"
    main(input_file_path, output_dir)
