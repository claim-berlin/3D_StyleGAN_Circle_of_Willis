import os
import sys
import math
import json
from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing
from contextlib import contextmanager, ExitStack
import numpy as np
import torch
from torch.backends import cudnn
from torch import nn, einsum
from torch.utils import data
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary
import torchio as tio
from einops import rearrange, repeat
from kornia.filters import filter2d, filter3d # added filter3d for blurring
import torchvision
from torchvision import transforms
from diff_augment import DiffAugment
# from vector_quantize_pytorch import VectorQuantize
from PIL import Image
from pathlib import Path
import nibabel as nib
from torch.cuda.amp import autocast, GradScaler
assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

import argparse
from retry.api import retry_call

# constants
NUM_CORES = multiprocessing.cpu_count()
EXTS = ['nii.gz', 'nii'] # changed extensions from jpg or png to nii.gz or nii

# make code faster when input size is constant
cudnn.benchmark = True

class NanException(Exception):
    pass

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


# Additonal feature related functions are commented out
# class RandomApply(nn.Module):
#     def __init__(self, prob, fn, fn_else = lambda x: x):
#         super().__init__()
#         self.fn = fn
#         self.fn_else = fn_else
#     def forward(self, x):
#         fn = self.fn if random() < self.prob else self.fn_else
#         return fn(x)
#
# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#     def forward(self, x):
#         return self.fn(x) + x
#
# class ChanNorm(nn.Module):
#     def __init__(self, dim, eps = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
#         self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))
#
#     def forward(self, x):
#         var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
#         mean = torch.mean(x, dim = 1, keepdim = True)
#         return (x - mean) / (var + self.eps).sqrt() * self.g + self.b
#
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = ChanNorm(dim)
#
#     def forward(self, x):
#         return self.fn(self.norm(x))

class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, *_, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[:, None, None] * f[None, :, None] * f[None, None, :]
        f = f.unsqueeze(0)
        # Apply the 3D filter to x
        return filter3d(x, f, normalized=True)


# # attention
# class DepthWiseConv2d(nn.Module):
#     def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv3d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias), # Changed to Conv3d
#             nn.Conv3d(dim_in, dim_out, kernel_size = 1, bias = bias) # Changed to Conv3d
#         )
#     def forward(self, x):
#         return self.net(x)
#
# class LinearAttention(nn.Module):
#     def __init__(self, dim, dim_head = 64, heads = 8):
#         super().__init__()
#         self.scale = dim_head ** -0.5
#         self.heads = heads
#         inner_dim = dim_head * heads
#
#         self.nonlin = nn.GELU()
#         self.to_q = nn.Conv3d(dim, inner_dim, 1, bias = False) # Changed to Conv3d
#         self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)
#         self.to_out = nn.Conv3d(inner_dim, dim, 1) # Changed to Conv3d
#
#     def forward(self, fmap):
#         h, x, y = self.heads, *fmap.shape[-2:]
#         q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
#         q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))
#
#         q = q.softmax(dim = -1)
#         k = k.softmax(dim = -2)
#
#         q = q * self.scale
#
#         context = einsum('b n d, b n e -> b d e', k, v)
#         out = einsum('b n d, b d e -> b n e', q, context)
#         out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)
#
#         out = self.nonlin(out)
#         return self.to_out(out)
#
# # one layer of self-attention and feedforward, for images
# attn_and_ff = lambda chan: nn.Sequential(*[
#     Residual(PreNorm(chan, LinearAttention(chan))),
#     Residual(PreNorm(chan, nn.Sequential(nn.Conv3d(chan, chan * 2, 1), leaky_relu(), nn.Conv3d(chan * 2, chan, 1)))) # Changed to Conv3d
# ])

# helpers
def exists(val):
    return val is not None

@contextmanager
def null_context():
    yield

def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]
    return multi_contexts

def default(value, d):
    return value if exists(value) else d

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def cast_list(el):
    return el if isinstance(el, list) else [el]

def parse_list(arg_value):
    return [item.strip() for item in arg_value.strip("[]").split(',')]

def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return not exists(t)

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts =  head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield

# For MedicalNet feature extraction
def generate_model():
    model = resnet10(sample_input_W=image_size[0],
                sample_input_H=image_size[1],
                sample_input_D=image_size[2],
                num_seg_classes=1)
    return model

def loss_backwards(fp16, scaler, loss, **kwargs):
    if fp16:
        scaler.scale(loss).backward(**kwargs)
    else:
        loss.backward(**kwargs)

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def calc_pl_lengths(styles, images):
    device = images.device
    # print("Shape of images while calculating path length penalty: ", images.shape)
    # print("Shape of styles while calculating path length penalty: ", styles.shape)
    num_voxels = images.shape[2] * images.shape[3] * images.shape[4]  # Adjusted for 3D
    # pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_voxels)
    pl_noise = torch.randn(images.shape, device=device) / (num_voxels ** (1/3))
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape, device=device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).cuda(device)

def noise_list(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]

def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)

def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, im_size, 1).uniform_(0., 1.).cuda(device)

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

# losses
def gen_hinge_loss(fake, real):
    return fake.mean()

def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

    def loss_half(t1, t2):
        t1 = rearrange(t1, 'i -> i ()')
        t2 = repeat(t2, 'j -> i j', i = t1.shape[0])
        t = torch.cat((t1, t2), dim = -1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device = device, dtype = torch.long))

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)


# dataset
def normalize_and_unsqueeze_nifti(image):
    # Normalize the image to the range [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image = np.transpose(image, (2, 0, 1))

    # Convert to a PyTorch tensor and add a channel dimension for grayscale
    image_grayscale_tensor = torch.from_numpy(image).unsqueeze(0).float()  # Shape: [1, D, H, W]
    return image_grayscale_tensor


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, transparent = False, aug_prob = 0.):
        super().__init__()
        # print(aug_prob)
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        self.transform = transforms.Compose([
            transforms.Lambda(normalize_and_unsqueeze_nifti),
            tio.RandomFlip(axes=(1,), flip_probability=aug_prob)  # horizontal flipping of real images
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = nib.load(path)
        data = img.get_fdata()

        return self.transform(data)

# augmentations
def random_hflip(tensor, prob):
    if prob < random():
        return tensor

    # horizontal flipping in differentiable augmentations
    return torch.flip(tensor, dims=(3,))

class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob = 0., types = [], detach = False):
        if random() < prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types=types)

        if detach:
            images = images.detach()

        return self.D(images)

# Stylegan2 classes
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class GrayscaleBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba=False, xy_upsample=False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 1  # Since images are grayscale
        self.conv = Conv3DMod(input_channel, out_filters, 1, demod=False)  # Using 3D convolution

        # Conditional upsampling
        if xy_upsample:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
                Blur()
            ) if upsample else None
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                Blur()
            ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, d, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_rgb):
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        return x


class Conv3DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        # Adjust the weight parameter initialization based on kernel dimensionality
        if isinstance(kernel, tuple):
            self.weight = nn.Parameter(torch.randn(out_chan, in_chan, *kernel))  # Unpack the tuple
        else:
            self.weight = nn.Parameter(torch.randn(out_chan, in_chan, kernel, kernel, kernel))
        # added another kernel for the 3rd dimension
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')


    def _get_same_padding(self, size, kernel, dilation, stride):
        if isinstance(kernel, tuple):
            # Calculate padding for each dimension separately
            padding = tuple(((s - 1) * (stride - 1) + dilation * (k - 1)) // 2 for s, k in zip(size, kernel))
            return padding
        else:
            # Single-dimensional kernel
            return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, depth, h, w = x.shape # Added depth dimension 'depth'

        w1 = y[:, None, :, None, None, None]
        w2 = self.weight[None, :, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4, 5), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, depth, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        # Adjust the call to _get_same_padding
        if isinstance(self.kernel, tuple):
            padding = self._get_same_padding((depth, h, w), self.kernel, self.dilation, self.stride)
        else:
            padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)


        x = F.conv3d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, depth, h, w)
        return x


class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample=True, upsample_rgb=True, rgba=False, xy_upsample=False, xy_upsample_rgb=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear',
                                    align_corners=False) if upsample and not xy_upsample else None

        # XY-specific 2D upsampling, Z or depth dimension remains the same
        self.xy_upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear',
                                       align_corners=False) if xy_upsample else None

        kernel = (3, 3, 3) if xy_upsample else 3
        #print(kernel)
        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv3DMod(input_channels, filters, kernel)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv3DMod(filters, filters, kernel)

        self.activation = leaky_relu()
        self.to_grayscale = GrayscaleBlock(latent_dim, filters, upsample_rgb, rgba, xy_upsample=xy_upsample_rgb)

    def forward(self, x, prev_rgb, istyle, inoise):
        if self.upsample is not None:
            x = self.upsample(x)

        elif self.xy_upsample is not None:
            x = self.xy_upsample(x)

        a = x.shape[2]
        b = x.shape[3]
        c = x.shape[4]

        inoise = inoise[:, :a, : b, : c, :]

        noise1 = self.to_noise1(inoise).permute((0, 4, 1, 2, 3))

        noise2 = self.to_noise2(inoise).permute((0, 4, 1, 2, 3))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)

        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_grayscale(x, prev_rgb, istyle)

        return x, rgb

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True, xy_specific=False):
        super().__init__()
        self.xy_specific = xy_specific
        stride = (1, 2, 2) if xy_specific else (2, 2, 2) if downsample else 1
        self.conv_res = nn.Conv3d(input_channels, filters, 1, stride = stride) # changed to 3d

        self.net = nn.Sequential(
            nn.Conv3d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv3d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        if downsample:
            conv_stride = (1, 2, 2) if xy_specific else (2, 2, 2)
            self.downsample = nn.Sequential(
                Blur(),
                nn.Conv3d(filters, filters, 3, padding=1, stride=conv_stride)
            )
        else:
            self.downsample = None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x

class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512): # 512 fmap_max
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)  # Layers for XY plane
        self.num_layers_z = int(log2(32) - 2)  # Layers for Z dimension # 32 for the Paper, please adapt depth parameter here.

        # the input parameter network-capacity is used to determine the number of filters for the generator blocks.
        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)

        filters = list(map(set_fmap_max, filters))
        print("Generator", filters)

        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])

        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4, 4)))# added 4 for the depths dimension
            #print(self.initial_block.shape)

        self.initial_conv = nn.Conv3d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])


        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            xy_upsample = ind > self.num_layers_z # Apply XY-specific upsampling in the later layers
            xy_upsample_rgb = ind >= self.num_layers_z  # Apply XY-s

            attn_fn = attn_and_ff(in_chan) if ind in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_rgb=not_last,
                rgba=transparent,
                xy_upsample = xy_upsample,
                xy_upsample_rgb = xy_upsample_rgb
            )
            self.blocks.append(block)


    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)
        i=0

        for style, block, attn in zip(styles, self.blocks, self.attns):

            if exists(attn):
                x = attn(x)

            x, rgb = block(x, rgb, style, input_noise)
            i+=1

        return rgb

class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity = 16, fq_layers = [], fq_dict_size = 256, attn_layers = [], transparent = False, fmap_max = 512):# 512
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        image_size_z = 32 # Layers for Z dimension # 32 for the Paper, please adapt depth parameter here.
        num_layers_z = int(log2(image_size_z) - 1)

        num_init_filters = 1

        blocks = []
        # the input parameter network-capacity is used to determine the number of filters for the discriminator blocks.
        filters = [num_init_filters] + [(network_capacity * 4) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))

        print("Discriminator", filters)
        chan_in_out = list(zip(filters[:-1], filters[1:]))
        print("Chan in out", chan_in_out)

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            is_not_last = ind != (len(chan_in_out) - 1)

            if ind < num_layers_z:
                block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            else:
                block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last, xy_specific=True)

            blocks.append(block)

            #Attention and quantization
            attn_fn = attn_and_ff(out_chan) if ind + 1 in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if ind + 1 in fq_layers else None
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * 2 * chan_last

        self.final_conv = nn.Conv3d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
            x = block(x)

            if exists(attn_block):
                x = attn_block(x)

            if exists(q_block):
                x, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)

        x = self.flatten(x)

        x = self.to_logit(x)
        return x.squeeze(), quantize_loss

class StyleGAN2(nn.Module):
    def __init__(self, image_size, latent_dim = 512, fmap_max = 512, style_depth = 8, network_capacity = 16, transparent = False, fp16 = False, cl_reg = False, steps = 1, lr = 1e-4, ttur_mult = 2, fq_layers = [], fq_dict_size = 256, attn_layers = [], no_const = False, lr_mlp = 0.1, rank = 0): # fmap_max 512
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        self.S = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const, fmap_max = fmap_max)
        self.D = Discriminator(image_size, network_capacity, fq_layers = fq_layers, fq_dict_size = fq_dict_size, attn_layers = attn_layers, transparent = transparent, fmap_max = fmap_max)

        self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const, fmap_max = fmap_max )

        self.D_cl = None

        if cl_reg:
            from contrastive_learner import ContrastiveLearner
            # experimental contrastive loss discriminator regularization
            assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size)

        # turn off grad for exponential moving averages
        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        # init optimizers
        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = Adam(generator_params, lr = self.lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = self.lr * ttur_mult, betas=(0.5, 0.9))

        # init weights
        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda(rank)

        # mixed precision
        self.fp16 = fp16

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv3d, nn.Linear}: # changed to Conv3d
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x

class Trainer():
    def __init__(
        self,
        name = 'default',
        results_dir = 'results',
        models_dir = 'models',
        base_dir = './',
        image_size = 128,
        network_capacity = 16,
        fmap_max = 512, #512
        transparent = False,
        batch_size = 4,
        mixed_prob = 0.9,
        gradient_accumulate_every=1,
        lr = 2e-4,
        lr_mlp = 0.1,
        ttur_mult = 2,
        rel_disc_loss = False,
        num_workers = None,
        save_every = 1000,
        evaluate_every = 1000,
        num_image_tiles = 8,
        trunc_psi = 0.6,
        fp16 = False,
        cl_reg = False,
        no_pl_reg = False,
        fq_layers = [],
        fq_dict_size = 256,
        attn_layers = [],
        no_const = False,
        aug_prob = 0.,
        aug_types = ['translation', 'cutout'],
        top_k_training = False,
        generator_top_k_gamma = 0.99,
        generator_top_k_frac = 0.5,
        dual_contrast_loss = False,
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        calculate_fid_num_images = 12800,
        clear_fid_cache = False,
        is_ddp = False,
        rank = 0,
        world_size = 1,
        log = False,
        *args,
        **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = base_dir / 'fid' / name
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.fmap_max = fmap_max
        self.transparent = transparent

        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size
        self.has_fq = len(self.fq_layers) > 0

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.rel_disc_loss = rel_disc_loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.num_image_tiles = num_image_tiles
        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.no_pl_reg = no_pl_reg
        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        self.fp16 = fp16
        self.scaler = GradScaler()

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.q_loss = None
        self.last_gp_loss = None
        self.last_cr_loss = None
        self.last_fid = None

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        self.top_k_training = top_k_training
        self.generator_top_k_gamma = generator_top_k_gamma
        self.generator_top_k_frac = generator_top_k_frac

        self.dual_contrast_loss = dual_contrast_loss

        assert not (is_ddp and cl_reg), 'Contrastive loss regularization does not work well with multi GPUs yet'
        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

        self.logger = aim.Session(experiment=name) if log else None

        # mixed precision
        self.scaler = GradScaler()

    @property
    def image_extension(self):
        return 'png' # if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    @property
    def hparams(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity}

    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr = self.lr, lr_mlp = self.lr_mlp, ttur_mult = self.ttur_mult, image_size = self.image_size, network_capacity = self.network_capacity, fmap_max = self.fmap_max, transparent = self.transparent, fq_layers = self.fq_layers, fq_dict_size = self.fq_dict_size, attn_layers = self.attn_layers, fp16 = self.fp16, cl_reg = self.cl_reg, no_const = self.no_const, rank = self.rank, *args, **kwargs)

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank]}
            self.S_ddp = DDP(self.GAN.S, **ddp_kwargs)
            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

        if exists(self.logger):
            self.logger.set_params(self.hparams)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']
        self.attn_layers = config.pop('attn_layers', [])
        self.no_const = config.pop('no_const', False)
        self.lr_mlp = config.pop('lr_mlp', 0.1)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'lr_mlp': self.lr_mlp, 'transparent': self.transparent, 'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers, 'no_const': self.no_const}

    def set_data_src(self, folder):
        self.dataset = Dataset(folder, self.image_size, transparent = self.transparent, aug_prob = self.dataset_aug_prob)
        num_workers = num_workers = default(self.num_workers, NUM_CORES if not self.is_ddp else 0)
        sampler = DistributedSampler(self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        dataloader = data.DataLoader(self.dataset, num_workers = num_workers, batch_size = math.ceil(self.batch_size / self.world_size), sampler = sampler, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)

        # auto set augmentation prob for user if dataset is detected to be low
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def train(self):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.tensor(0.).cuda(self.rank)
        total_gen_loss = torch.tensor(0.).cuda(self.rank)

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        aug_prob   = self.aug_prob
        aug_types  = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        apply_gradient_penalty = self.steps % 16 == 0
        apply_path_penalty = not self.no_pl_reg and self.steps > 5000 and self.steps % 32 == 0
        apply_cl_reg_to_generated = self.steps > 20000

        S = self.GAN.S if not self.is_ddp else self.S_ddp
        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        backwards = partial(loss_backwards, self.fp16)

        if exists(self.GAN.D_cl):
            self.GAN.D_opt.zero_grad()

            if apply_cl_reg_to_generated:
                for i in range(self.gradient_accumulate_every):
                    get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
                    style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)
                    noise = image_noise(batch_size, image_size, device=self.rank)

                    w_space = latent_to_w(self.GAN.S, style)
                    w_styles = styles_def_to_tensor(w_space)

                    generated_images = self.GAN.G(w_styles, noise)
                    self.GAN.D_cl(generated_images.clone().detach(), accumulate=True)

            for i in range(self.gradient_accumulate_every):
                image_batch = next(self.loader).cuda(self.rank)
                self.GAN.D_cl(image_batch, accumulate=True)

            loss = self.GAN.D_cl.calculate_loss()
            self.last_cr_loss = loss.clone().detach().item()
            backwards(loss, self.GAN.D_opt, loss_id = 0)

            self.GAN.D_opt.step()

        # setup losses
        if not self.dual_contrast_loss:
            D_loss_fn = hinge_loss
            G_loss_fn = gen_hinge_loss
            G_requires_reals = False
        else:
            D_loss_fn = dual_contrastive_loss
            G_loss_fn = dual_contrastive_loss
            G_requires_reals = True

        # train discriminator
        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, S, G]):
            get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(batch_size, image_size, device=self.rank)

            w_space = latent_to_w(S, style)
            w_styles = styles_def_to_tensor(w_space)

            with autocast(enabled=self.fp16):
                generated_images = G(w_styles, noise)
                fake_output, fake_q_loss = D_aug(generated_images.clone().detach(), detach = True, **aug_kwargs)

                image_batch = next(self.loader).cuda(self.rank)
                image_batch.requires_grad_()
                real_output, real_q_loss = D_aug(image_batch, **aug_kwargs)

                real_output_loss = real_output
                fake_output_loss = fake_output

                if self.rel_disc_loss:
                    real_output_loss = real_output_loss - fake_output.mean()
                    fake_output_loss = fake_output_loss - real_output.mean()

                divergence = D_loss_fn(real_output_loss, fake_output_loss)
                disc_loss = divergence

                if self.has_fq:
                    quantize_loss = (fake_q_loss + real_q_loss).mean()
                    self.q_loss = float(quantize_loss.detach().item())

                    disc_loss = disc_loss + quantize_loss

                if apply_gradient_penalty:
                    gp = gradient_penalty(image_batch, real_output)
                    self.last_gp_loss = gp.clone().detach().item()
                    self.track(self.last_gp_loss, 'GP')
                    disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(self.scaler, disc_loss)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.track(self.d_loss, 'D')

        if self.fp16:
            self.scaler.step(self.GAN.D_opt)
            self.scaler.update()
        else:
            self.GAN.D_opt.step()

        # train generator
        self.GAN.G_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[S, G, D_aug]):
            style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(batch_size, image_size, device=self.rank)

            w_space = latent_to_w(S, style)
            w_styles = styles_def_to_tensor(w_space)
            with autocast(enabled=self.fp16):
                generated_images = G(w_styles, noise)

                fake_output, _ = D_aug(generated_images, **aug_kwargs)
                fake_output_loss = fake_output

                real_output = None
                if G_requires_reals:
                    image_batch = next(self.loader).cuda(self.rank)
                    real_output, _ = D_aug(image_batch, detach = True, **aug_kwargs)
                    real_output = real_output.detach()

                if self.top_k_training:
                    epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
                    k_frac = max(self.generator_top_k_gamma ** epochs, self.generator_top_k_frac)
                    k = math.ceil(batch_size * k_frac)

                    if k != batch_size:
                        fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)

                loss = G_loss_fn(fake_output_loss, real_output)
                gen_loss = loss

                if apply_path_penalty:
                    pl_lengths = calc_pl_lengths(w_styles, generated_images)
                    avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                    if not is_empty(self.pl_mean):
                        pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                        if not torch.isnan(pl_loss):
                            gen_loss = gen_loss + pl_loss

                gen_loss = gen_loss / self.gradient_accumulate_every
                gen_loss.register_hook(raise_if_nan)
                backwards(self.scaler, gen_loss)

                total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.track(self.g_loss, 'G')

        if self.fp16:
            self.scaler.step(self.GAN.G_opt)
            self.scaler.update()
        else:
            self.GAN.G_opt.step()

        # calculate moving averages
        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
            self.track(self.pl_mean, 'PL')

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors
        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        # periodically save results
        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 2500):
                self.evaluate(floor(self.steps / self.evaluate_every))
                # Save model architectures at the beginning of training
                os.makedirs(str(self.results_dir / self.name / "SUMMARY"), exist_ok=True)
                with open(str(self.results_dir / self.name / "SUMMARY/model_structures.txt"), 'w') as file:
                    discriminator_summary = summary(D_aug, input_size=(32, 1, 32, image_size, image_size))
                    file.write(str(discriminator_summary))

                    print("\n\n-----------------------\n\n", file=file)

                    # Print and write the Generator structure
                    generator_summary = summary(G, input_data=(w_styles, noise))

                    file.write(str(generator_summary))

            if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                num_batches = math.ceil(self.calculate_fid_num_images / self.batch_size)
                fid, medical_FID, prd_AUC = self.calculate_fid(num_batches)

                self.last_fid = fid

                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'Step - {self.steps} -- FID: {fid} -- MD: {medical_FID} -- prd_AUC: {prd_AUC}\n')

        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, num = 0, trunc = 1.0):
        print("Evaluating")

        self.GAN.eval()
        ext = self.image_extension
        num_rows = self.num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        # regular

        generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n, trunc_psi = self.trunc_psi)

        # print("---------------------------------------------------", generated_images.shape)

        generated_images = generated_images[:, :, 15, :, :]
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)

        # moving averages

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
        generated_images = generated_images[:, :, 15, :, :]
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)

        # mixing regularities

        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda(self.rank)
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim, device=self.rank)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, trunc_psi = self.trunc_psi)
        generated_images = generated_images[:, :, 15, :, :]
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'), nrow=num_rows)

    @torch.no_grad()
    def calculate_fid(self, num_batches):
        from pytorch_fid import fid_score
        sys.path.append(str(Path(__file__).resolve().parent))
        from MedicalNet.MFD_PRD import generate_model, process_dataset, load_features_from_npy_gz, compute_prd_from_embedding, calculate_Medical_fid
        from MedicalNet.dataset import GANDataset
        from MedicalNet.resnet import resnet10
        torch.cuda.empty_cache()

        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        real_path_3d = self.fid_dir / 'real_3d'
        fake_path_3d = self.fid_dir / 'fake_3d'

        # remove any existing files used for fid calculation and recreate directories

        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)
            rmtree(real_path_3d, ignore_errors=True)
            os.makedirs(real_path_3d)

            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                real_batch = next(self.loader)

                for k, image in enumerate(real_batch.unbind(0)):
                    # print(image.shape)
                    nifti = nib.Nifti1Image(image.squeeze(0).cpu().detach().numpy(), np.eye(4))
                    nib.save(nifti, real_path_3d / f'{str(k + batch_num * self.batch_size)}.nii.gz')

                    for i in range(image.shape[1]):
                        image_slice = image[:, i, :, :]
                        filename = str(k + batch_num * self.batch_size)+f"-{i}"
                        torchvision.utils.save_image(image_slice, str(real_path / f'{filename}.png'))

        # generate a bunch of fake images in results / name / fid_fake
        rmtree(fake_path, ignore_errors=True)
        rmtree(fake_path_3d, ignore_errors=True)
        os.makedirs(fake_path)
        os.makedirs(fake_path_3d)

        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # Number of GPUs available. Use 0 for CPU mode.
        ngpu = 1
        # gpu number
        cuda_n = [0]

        device = torch.device("cuda:" + str(cuda_n[0]) if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = noise_list(self.batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(self.batch_size, image_size, device=self.rank)

            # moving averages
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, noise, trunc_psi = self.trunc_psi)

            for j, image in enumerate(generated_images.unbind(0)):
                nifti = nib.Nifti1Image(image.squeeze(0).cpu().detach().numpy(), np.eye(4))#
                nib.save(nifti, fake_path_3d / f'{str(j + batch_num * self.batch_size)}.nii.gz')

                for i in range(image.shape[1]):
                    image_slice = image[:, i, :, :]
                    torchvision.utils.save_image(image_slice, str(fake_path / f'{str(j + batch_num * self.batch_size)}-f{i}-ema.{ext}'))

        pretrained_model_path = os.path.join(self.base_dir, "resnet_10_23dataset.pth")
        # CUSTOM MEDICAL_NET BASED FID AND PRD CALCULATION
        # getting model
        print(f'getting pretrained model on {device} \n')
        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        net = generate_model()
        print('Resnet model created \n')

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu >= 1):
            net = nn.DataParallel(net, cuda_n)

        net.load_state_dict(checkpoint['state_dict'])
        print('Resnet model loaded with pretrained weights')

        # paths for real and generated features to be saved
        data_dir_real_features = os.path.join(self.base_dir, "MedicalNet", "features", "real")
        data_dir_gen_features = os.path.join(self.base_dir, "MedicalNet", "features", "fake")

        os.makedirs(data_dir_real_features, exist_ok = True)
        os.makedirs(data_dir_gen_features, exist_ok = True)

        print(f'Creating features from {fake_path_3d}')
        process_dataset(fake_path_3d, data_dir_gen_features, net)

        print(f'Creating features from {real_path_3d}')
        process_dataset(real_path_3d, data_dir_real_features, net)

        print("All features extracted with MedicalNet \n")

        real_features = load_features_from_npy_gz(data_dir_real_features)
        gen_features = load_features_from_npy_gz(data_dir_gen_features)

        # Calculate PRD
        precision, recall = compute_prd_from_embedding(gen_features, real_features, enforce_balance=False)
        sorted_indices = np.argsort(recall)
        sorted_recall = recall[sorted_indices]
        sorted_precision = precision[sorted_indices]

        # traditional FID calculation
        fid = fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, noise.device, 2048)

        # Medical Net based FID calculation
        fid_MedicalNET_score = calculate_Medical_fid(real_features, gen_features)

        # Calculate the AUC of PRD
        PRD_AUC = np.trapz(sorted_precision, sorted_recall)

        print("AUC of the PRD curve:", PRD_AUC)
        print(f'Medical Net based FID score: {fid_MedicalNET_score}')

        return fid, fid_MedicalNET_score, PRD_AUC

    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi = 0.75):
        S = self.GAN.S
        batch_size = self.batch_size
        latent_dim = self.GAN.G.latent_dim

        if not exists(self.av):
            z = noise(2000, latent_dim, device=self.rank)
            samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        av_torch = torch.from_numpy(self.av).cuda(self.rank)
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor

    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi = 0.75):
        w_space = []
        for tensor, num_layers in w:
            tensor = self.truncate_style(tensor, trunc_psi = trunc_psi)
            w_space.append((tensor, num_layers))
        return w_space

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8):
        w = map(lambda t: (S(t[0]), t[1]), style)
        w_truncated = self.truncate_style_defs(w, trunc_psi = trunc_psi)
        w_styles = styles_def_to_tensor(w_truncated)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, trunc = 1.0, num_steps = 100, save_frames = False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise
        latents_low = noise(num_rows ** 2, latent_dim, device=self.rank)
        latents_high = noise(num_rows ** 2, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        ratios = torch.linspace(0., 1., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for i in range(generated_images.shape[0]):  # Iterate over each volume
                single_volume = generated_images[i].squeeze().detach().cpu().numpy()  # Convert to numpy array
                nifti_img = nib.Nifti1Image(single_volume, np.eye(4))  # np.eye(4) is a placeholder for the affine matrix

                nifti_filename = os.path.join(folder_path, f'{str(num)}_{i:03d}.nii.gz')
                nib.save(nifti_img, nifti_filename)

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('PL', self.pl_mean),
            ('CR', self.last_cr_loss),
            ('Q', self.q_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def track(self, value, name):
        if not exists(self.logger):
            return
        self.logger.track(value, name = name)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
        }

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if 'version' in load_data:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e


class ModelLoader:
    def __init__(self, *, base_dir, name = 'default',fmap_max=512, network_capacity=16, load_from = -1):
        self.model = Trainer(name = name, base_dir = base_dir, fmap_max=fmap_max, network_capacity=network_capacity)
        self.model.load(load_from)
        fmap_max

    def noise_to_styles(self, noise, trunc_psi = None):
        noise = noise.cuda()
        w = self.model.GAN.SE(noise)

        if exists(trunc_psi):
            w = self.model.truncate_style(w, trunc_psi=trunc_psi)
        return w

    def styles_to_images(self, w):
        batch_size, *_ = w.shape
        num_layers = self.model.GAN.GE.num_layers
        image_size = self.model.image_size
        w_def = [(w, num_layers)]

        w_tensors = styles_def_to_tensor(w_def)
        noise = image_noise(batch_size, image_size, device = 0)

        images = self.model.GAN.GE(w_tensors, noise)
        images.clamp_(0., 1.)
        return images

    def styles_to_images_ext_noise(self, w, n):
        batch_size, *_ = w.shape
        num_layers = self.model.GAN.GE.num_layers
        image_size = self.model.image_size
        w_def = [(w, num_layers)]

        w_tensors = styles_def_to_tensor(w_def)

        images = self.model.GAN.GE(w_tensors, n)

        images.clamp_(0., 1.)
        return images


def cast_list(el):
    return el if isinstance(el, list) else [el]

def timestamped_filename(prefix = 'generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def run_training(rank, world_size, model_args, data, load_from, new, num_train_steps, name, seed):
    is_main = rank == 0
    is_ddp = world_size > 1

    if is_ddp:
        set_seed(seed)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        print(f"{rank + 1}/{world_size} process initialized.")

    model_args.update(
        is_ddp = is_ddp,
        rank = rank,
        world_size = world_size
    )

    model = Trainer(**model_args)

    if not new:
        model.load(load_from)
    else:
        model.clear()

    model.set_data_src(data)

    progress_bar = tqdm(initial = model.steps, total = num_train_steps, mininterval=10., desc=f'{name}<{data}>')
    while model.steps < num_train_steps:
        retry_call(model.train, tries=3, exceptions=NanException)
        progress_bar.n = model.steps
        progress_bar.refresh()
        if is_main and model.steps % 50 == 0:
            model.print_log()

    model.save(model.checkpoint_num)

    if is_ddp:
        dist.destroy_process_group()

def train_from_folder(
    data = './data',
    results_dir = './results',
    models_dir = './models',
    name = 'Medium_v1',
    new = True,
    load_from = -1,
    image_size = 128,
    network_capacity = 8,
    fmap_max = 512,
    transparent = False,
    batch_size = 8,
    gradient_accumulate_every = 4,
    num_train_steps = 100000,
    learning_rate = 1e-4,
    lr_mlp = 0.1,
    ttur_mult = 1.5,
    rel_disc_loss = False,
    num_workers =  None,
    save_every = 3000,
    evaluate_every = 3000,
    generate = False,
    num_generate = 1,
    generate_interpolation = False,
    interpolation_num_steps = 100,
    save_frames = False,
    num_image_tiles = 8,
    trunc_psi = 1,
    mixed_prob = 0.9,
    fp16 = True,
    no_pl_reg = False,
    cl_reg = False,
    fq_layers = [],
    fq_dict_size = 256,
    attn_layers = [],
    no_const = False,
    aug_prob = 0.5,
    aug_types = ['translation', 'cutout'],
    top_k_training = False,
    generator_top_k_gamma = 0.99,
    generator_top_k_frac = 0.5,
    dual_contrast_loss = False,
    dataset_aug_prob = 0.5,
    multi_gpus = False,
    calculate_fid_every = None,
    calculate_fid_num_images = 1782,
    clear_fid_cache = False,
    seed = 42,
    log = False
):
    model_args = dict(
        name = name,
        results_dir = results_dir,
        models_dir = models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        image_size = image_size,
        network_capacity = network_capacity,
        fmap_max = fmap_max,
        transparent = transparent,
        lr = learning_rate,
        lr_mlp = lr_mlp,
        ttur_mult = ttur_mult,
        rel_disc_loss = rel_disc_loss,
        num_workers = num_workers,
        save_every = save_every,
        evaluate_every = evaluate_every,
        num_image_tiles = num_image_tiles,
        trunc_psi = trunc_psi,
        fp16 = fp16,
        no_pl_reg = no_pl_reg,
        cl_reg = cl_reg,
        fq_layers = fq_layers,
        fq_dict_size = fq_dict_size,
        attn_layers = attn_layers,
        no_const = no_const,
        aug_prob = aug_prob,
        aug_types = aug_types,
        top_k_training = top_k_training,
        generator_top_k_gamma = generator_top_k_gamma,
        generator_top_k_frac = generator_top_k_frac,
        dual_contrast_loss = dual_contrast_loss,
        dataset_aug_prob = dataset_aug_prob,
        calculate_fid_every = calculate_fid_every,
        calculate_fid_num_images = calculate_fid_num_images,
        clear_fid_cache = clear_fid_cache,
        mixed_prob = mixed_prob,
        log = log
    )

    if generate:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        for num in tqdm(range(num_generate)):
            model.evaluate(f'{samples_name}-{num}', num_image_tiles)
        print(f'sample images generated at {results_dir}/{name}/{samples_name}')
        return

    if generate_interpolation:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        model.generate_interpolation(samples_name, num_image_tiles, num_steps = interpolation_num_steps, save_frames = save_frames)
        print(f'interpolation generated at {results_dir}/{name}/{samples_name}')
        return

    world_size = torch.cuda.device_count()

    if world_size == 1 or not multi_gpus:
        run_training(0, 1, model_args, data, load_from, new, num_train_steps, name, seed)
        return

    mp.spawn(run_training,
        args=(world_size, model_args, data, load_from, new, num_train_steps, name, seed),
        nprocs=world_size,
        join=True)

def main():
    parser = argparse.ArgumentParser(description='Train a model with StyleGAN2 PyTorch.')
    parser.add_argument('--data', type=str, default='./', help='Path to the data directory.')
    parser.add_argument('--results_dir', type=str, default='./results', help='Path to the results directory.')
    parser.add_argument('--models_dir', type=str, default='./models', help='Path to the models directory.')
    parser.add_argument('--name', type=str, default='default_Medium_v2', help='Name of the experiment.')
    parser.add_argument('--new', action='store_true', help='Flag to train a new model.')
    parser.add_argument('--image_size', type=int, default=128, help='Size of the images.')
    parser.add_argument('--fmap_max', type=int, default=512, help='Maximum number of convolutional filter maps')
    parser.add_argument('--network_capacity', type=int, default=8, help='defines network capacity by setting lowest number of convolutional filters for the discriinator(4x) and generator(2x)')
    parser.add_argument('--trunc_psi', type=float, default=1.0, help='truncation parameter psi usually between 0 and 1')
    parser.add_argument('--dataset_aug_prob', type=float, default=0.5, help='defines augmentation probability for real images (horizontal flipping)')
    parser.add_argument('--aug-types', type=parse_list, default=["translation", "cutout"], help='Types of differentiable augmentations: selection from translation, cutout, contrast')
    parser.add_argument('--aug-prob', type=float, default=0.5, help='differentiable augmentation probability')
    parser.add_argument('--ttur_mult', type=float, default=1.5, help='Two Time-scale Update Rule multiplier')
    parser.add_argument('--save_every', type=int, default=3000, help='Save model checkpoint every n steps')
    parser.add_argument('--calculate_fid_every', type=int, default=3000, help='Calculate FID, MD, AUC-PRD every n steps')
    parser.add_argument('--calculate_fid_num_images', type=int, default=1782, help='calculate metrics using n generated volumes')
    parser.add_argument('--fp16', type=bool, default=True, help='mixed precision')
    parser.add_argument('--gradient-accumulate-every', type=int, default=4, help='gradient accumulation for n times in each step')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=4e-05, help='learning rate of generator')
    parser.add_argument('--no_pl_reg', type=bool, default=False, help='whether to use path length regularization for the generator')
    parser.add_argument('--image-size', type=int, default=128, help='image size in width x heigth')

    args = parser.parse_args()

    kwargs = vars(args)

    train_from_folder(**kwargs)

if __name__ == "__main__":
    main()