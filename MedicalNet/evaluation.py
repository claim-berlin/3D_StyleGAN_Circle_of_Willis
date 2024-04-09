import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import resnet
from dataset import GANDataset
from scipy.linalg import sqrtm
import glob
import gzip
from io import BytesIO
import sklearn
from sklearn.cluster import MiniBatchKMeans
from pytorch_fid import fid_score
from joblib import Parallel, delayed
import os
import nibabel as nib
from PIL import Image
import numpy as np

# This evaluation script is adapted from https://github.com/prediction2020/3DGAN_synthesis_of_3D_TOF_MRA_with_segmentation_labels

def generate_model():
    model = resnet.resnet10(sample_input_W=image_size[0],
                            sample_input_H=image_size[1],
                            sample_input_D=image_size[2],
                            num_seg_classes=1)
    return model


def get_features(data_loader, model, save_features_dir):
    model.eval()
    for batch_id, batch_data in enumerate(tqdm(data_loader)):
        volume = batch_data
        volume = volume.to(device)

        with torch.no_grad():
            feature = model(volume).detach().cpu()

        if not isinstance(feature, np.ndarray):
            feature = np.asarray(feature)

        f = gzip.GzipFile(save_features_dir + "/" + str(batch_id) + '_em_features.npy.gz', "w")
        np.save(file=f, arr=feature)
        f.close()

    return


# Compute FID
def load_features_from_npy_gz(directory):
    """
    Load and flatten features from .npy.gz files in the given directory,
    ensuring no empty arrays and cleaning NaNs and Infs.
    """
    features = []
    # Iterate over .npy.gz files in the directory
    for file_path in glob.glob(os.path.join(directory, '*.npy.gz')):
        # print("Loading:", file_path)
        # Open the .npy.gz file with gzip
        with gzip.open(file_path, 'rb') as f:
            # Read the decompressed content into a buffer
            buf = BytesIO(f.read())
            # Load the buffer with np.load
            data = np.load(buf, allow_pickle=True)
            # Flatten the numpy array and append to features list
            feature_array = data.flatten()
            # Remove NaNs and Infs
            feature_array = feature_array[~np.isnan(feature_array)]
            feature_array = feature_array[np.isfinite(feature_array)]
            if feature_array.size > 0:
                features.append(feature_array)
    if len(features) == 0:
        raise ValueError("No valid feature data found in directory: " + directory)
    return np.asarray(features)

# Compute MedicalNet Distance
def calculate_Medical_fid(features_1, features_2):
    """
    Calculate the Fréchet Inception Distance (FID) between two sets of features.

    Parameters:
    features_1 (numpy.ndarray): Feature set 1.
    features_2 (numpy.ndarray): Feature set 2.

    Returns:
    float: The FID score.
    """
    mu1, sigma1 = features_1.mean(axis=0), np.cov(features_1, rowvar=False)
    mu2, sigma2 = features_2.mean(axis=0), np.cov(features_2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# PRD Computation
def compute_prd(eval_dist, ref_dist, num_angles=1001, epsilon=1e-10):
    """Computes the PRD curve for discrete distributions.

    This function computes the PRD curve for the discrete distribution eval_dist
    with respect to the reference distribution ref_dist. This implements the
    algorithm in [arxiv.org/abs/1806.2281349]. The PRD will be computed for an
    equiangular grid of num_angles values between [0, pi/2].

    Args:
      eval_dist: 1D NumPy array or list of floats with the probabilities of the
                 different states under the distribution to be evaluated.
      ref_dist: 1D NumPy array or list of floats with the probabilities of the
                different states under the reference distribution.
      num_angles: Number of angles for which to compute PRD. Must be in [3, 1e6].
                  The default value is 1001.
      epsilon: Angle for PRD computation in the edge cases 0 and pi/2. The PRD
               will be computes for epsilon and pi/2-epsilon, respectively.
               The default value is 1e-10.

    Returns:
      precision: NumPy array of shape [num_angles] with the precision for the
                 different ratios.
      recall: NumPy array of shape [num_angles] with the recall for the different
              ratios.

    Raises:
      ValueError: If not 0 < epsilon <= 0.1.
      ValueError: If num_angles < 3.
    """

    if not (epsilon > 0 and epsilon < 0.1):
        raise ValueError('epsilon must be in (0, 0.1] but is %s.' % str(epsilon))
    if not (num_angles >= 3 and num_angles <= 1e6):
        raise ValueError('num_angles must be in [3, 1e6] but is %d.' % num_angles)

    # Compute slopes for linearly spaced angles between [0, pi/2]
    angles = np.linspace(epsilon, np.pi / 2 - epsilon, num=num_angles)
    slopes = np.tan(angles)

    # Broadcast slopes so that second dimension will be states of the distribution
    slopes_2d = np.expand_dims(slopes, 1)

    # Broadcast distributions so that first dimension represents the angles
    ref_dist_2d = np.expand_dims(ref_dist, 0)
    eval_dist_2d = np.expand_dims(eval_dist, 0)

    # Compute precision and recall for all angles in one step via broadcasting
    precision = np.minimum(ref_dist_2d * slopes_2d, eval_dist_2d).sum(axis=1)
    recall = precision / slopes

    # handle numerical instabilities leaing to precision/recall just above 1
    max_val = max(np.max(precision), np.max(recall))
    if max_val > 1.001:
        raise ValueError('Detected value > 1.001, this should not happen.')
    precision = np.clip(precision, 0, 1)
    recall = np.clip(recall, 0, 1)

    return precision, recall


def compute_prd_from_embedding(eval_data, ref_data, num_clusters=10,
                               num_angles=1001, num_runs=10,
                               enforce_balance=True):
    """Computes PRD data from sample embeddings.

    The points from both distributions are mixed and then clustered. This leads
    to a pair of histograms of discrete distributions over the cluster centers
    on which the PRD algorithm is executed.

    The number of points in eval_data and ref_data must be equal since
    unbalanced distributions bias the clustering towards the larger dataset. The
    check can be disabled by setting the enforce_balance flag to False (not
    recommended).

    Args:
      eval_data: NumPy array of data points from the distribution to be evaluated.
      ref_data: NumPy array of data points from the reference distribution.
      num_clusters: Number of cluster centers to fit. The default value is 20.
      num_angles: Number of angles for which to compute PRD. Must be in [3, 1e6].
                  The default value is 1001.
      num_runs: Number of independent runs over which to average the PRD data.
      enforce_balance: If enabled, throws exception if eval_data and ref_data do
                       not have the same length. The default value is True.

    Returns:
      precision: NumPy array of shape [num_angles] with the precision for the
                 different ratios.
      recall: NumPy array of shape [num_angles] with the recall for the different
              ratios.

    Raises:
      ValueError: If len(eval_data) != len(ref_data) and enforce_balance is set to
                  True.
    """

    if enforce_balance and len(eval_data) != len(ref_data):
        raise ValueError(
            'The number of points in eval_data %d is not equal to the number of '
            'points in ref_data %d. To disable this exception, set enforce_balance '
            'to False (not recommended).' % (len(eval_data), len(ref_data)))

    eval_data = np.array(eval_data, dtype=np.float64)
    ref_data = np.array(ref_data, dtype=np.float64)
    precisions = []
    recalls = []

    for _ in range(num_runs):
        eval_dist, ref_dist = _cluster_into_bins(eval_data, ref_data, num_clusters)
        precision, recall = compute_prd(eval_dist, ref_dist, num_angles)
        precisions.append(precision)
        recalls.append(recall)
    precision = np.mean(precisions, axis=0)
    recall = np.mean(recalls, axis=0)

    return precision, recall


def _cluster_into_bins(eval_data, ref_data, num_clusters):
    """Clusters the union of the data points and returns the cluster distribution.

    Clusters the union of eval_data and ref_data into num_clusters using minibatch
    k-means. Then, for each cluster, it computes the number of points from
    eval_data and ref_data.

    Args:
      eval_data: NumPy array of data points from the distribution to be evaluated.
      ref_data: NumPy array of data points from the reference distribution.
      num_clusters: Number of cluster centers to fit.

    Returns:
      Two NumPy arrays, each of size num_clusters, where i-th entry represents the
      number of points assigned to the i-th cluster.
    """

    cluster_data = np.vstack([eval_data, ref_data])
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, n_init=10)
    labels = kmeans.fit(cluster_data).labels_

    eval_labels = labels[:len(eval_data)]
    ref_labels = labels[len(eval_data):]

    eval_bins = np.histogram(eval_labels, bins=num_clusters,
                             range=[0, num_clusters], density=True)[0]
    ref_bins = np.histogram(ref_labels, bins=num_clusters,
                            range=[0, num_clusters], density=True)[0]
    return eval_bins, ref_bins  # , silhouette, curr_sse


def process_dataset(dataroot, save_features_dir, net):
    dataset = GANDataset(dataroot)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    # Assuming get_features is defined elsewhere and correctly processes the data_loader and net
    get_features(data_loader, net, save_features_dir)


def convert_nifti_to_png(file_path, output_directory):
    # Load the NIfTI file
    img = nib.load(file_path)
    data = img.get_fdata()

    # Determine the number of slices (assuming the slices are along the z-axis)
    num_slices = data.shape[2]

    # Limit the number of slices to process to 32
    num_slices = min(num_slices, 32)  # change this number

    # Process each slice and save as a new file
    for i in range(num_slices):
        slice_data = data[:, :, i]
        # Normalize the data to the range of the data type
        norm_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
        norm_data = (255 * norm_data).astype(np.uint8)

        # Convert to PIL Image and save
        img = Image.fromarray(norm_data, mode='L')
        output_path = os.path.join(output_directory, os.path.basename(file_path).replace('.nii.gz', f'_{i + 1}.png'))
        img.save(output_path)


def convert_directory_to_2d_slices(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get all NIfTI files in the directory
    nifti_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.nii.gz')]

    # Process each file in parallel
    Parallel(n_jobs=-1)(delayed(convert_nifti_to_png)(file_path, output_directory) for file_path in nifti_files)


def squeeze_and_normalize_nifti(input_nifti_path):
    """
    squeezes and normalizes a given nifti file path and overwrites the file

    Parameters:
    - input_nifti_path: str, the file path of the input NIfTI image.
    - output_nifti_path: str, the file path where the squeezed and normalized image will be saved.
    """
    # Load the NIfTI file
    nifti_img = nib.load(input_nifti_path)

    # Get the image data array
    img_data = nifti_img.get_fdata()

    if 1 in img_data.shape:
        img_data = np.squeeze(img_data)

        # Normalize the image data
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
    else:
        # Normalize the image data without squeezing
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

    # Create a new NIfTI image
    squeezed_nifti = nib.Nifti1Image(img_data, np.eye(4), nifti_img.header)

    # Save the image
    nib.save(squeezed_nifti, input_nifti_path)


# Evaluation directory
eval_dir = "/home/user/EVALUATION"  # adjust the path

# Pretrained MedicalNet path
pretrained_model_path = "./resnet_10_23dataset.pth"  # adjust the path

image_size = (32, 128, 128)
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# gpu number
cuda_n = [0]

device = torch.device("cuda:" + str(cuda_n[0]) if (torch.cuda.is_available() and ngpu > 0) else "cpu")

print(f'getting pretrained model on {device} \n')
checkpoint = torch.load(pretrained_model_path, map_location='cpu')
# print(checkpoint['state_dict'].keys())
net = generate_model()
print('Resnet model created \n')

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu >= 1):
    net = nn.DataParallel(net, cuda_n)

net.load_state_dict(checkpoint['state_dict'])
print('Resnet model loaded with pretrained weights')

# data paths
data_dir_real_3D = os.path.join(eval_dir, "real_subset_1")
data_dir_gen_3D = os.path.join(eval_dir, "real_subset_2")

data_dir_real_2D = data_dir_real_3D + "_2D"
data_dir_gen_2D = data_dir_gen_3D + "_2D"

# paths for real and generated features to be saved
data_dir_real_features = os.path.join(eval_dir, "MD_features/real")
data_dir_gen_features = os.path.join(eval_dir, "MD_features/gen")

# List of all NIfTI files in the directory
nifti_files = [os.path.join(data_dir_gen_3D, f) for f in os.listdir(data_dir_gen_3D) if f.endswith(".nii.gz")]
Parallel(n_jobs=-1)(delayed(squeeze_and_normalize_nifti)(file_path) for file_path in nifti_files)

nifti_files = [os.path.join(data_dir_real_3D, f) for f in os.listdir(data_dir_real_3D) if f.endswith(".nii.gz")]
Parallel(n_jobs=-1)(delayed(squeeze_and_normalize_nifti)(file_path) for file_path in nifti_files)

convert_directory_to_2d_slices(data_dir_real_3D, data_dir_real_2D)
convert_directory_to_2d_slices(data_dir_gen_3D, data_dir_gen_2D)

os.makedirs(data_dir_real_features, exist_ok=True)
os.makedirs(data_dir_gen_features, exist_ok=True)

print(f'Creating features from {data_dir_gen_3D}')
process_dataset(data_dir_gen_3D, data_dir_gen_features, net)

print(f'Creating features from {data_dir_real_3D}')
process_dataset(data_dir_real_3D, data_dir_real_features, net)

print("All features extracted with MedicalNet \n")

real_features = load_features_from_npy_gz(data_dir_real_features)
gen_features = load_features_from_npy_gz(data_dir_gen_features)

# Medical Net based FID calculation
fid_MedicalNET_score = calculate_Medical_fid(real_features, gen_features)

# Calculate PRD
precision, recall = compute_prd_from_embedding(gen_features, real_features)
sorted_indices = np.argsort(recall)
sorted_recall = recall[sorted_indices]
sorted_precision = precision[sorted_indices]

# Calculate the AUC of PRD
auc_prd = np.trapz(sorted_precision, sorted_recall)

print("AUC of the PRD curve:", auc_prd)
print(f'Medical Net based FID score: {fid_MedicalNET_score}')

fid = fid_score.calculate_fid_given_paths([str(data_dir_real_2D), str(data_dir_gen_2D)], 256, device, 2048)

print(f'FID score: {fid}')

output_directory = eval_dir
metrics_file_path = os.path.join(output_directory, 'metric_values.txt')

with open(metrics_file_path, 'w') as file:
    file.write(f"AUC of the PRD curve: {auc_prd}\n")
    file.write(f"Medical Net based FID score: {fid_MedicalNET_score}\n")
    file.write(f"FID score: {fid}\n")

print(f"Metrics saved to {metrics_file_path}")
