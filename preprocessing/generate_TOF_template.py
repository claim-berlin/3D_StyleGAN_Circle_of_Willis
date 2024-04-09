import subprocess
import os
from joblib import Parallel, delayed

# This script generates a custom TOF MRA template given a dataset and a reference image and follows the protocol provided in the below link.
# https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/FAQ


def run_flirt(in_image, ref_image, out_image, matrix_out):
    """Run FLIRT to register an image to the reference."""
    print(f"Registering {in_image} to {ref_image}...")
    flirt_cmd = f"flirt -in {in_image} -ref {ref_image} -dof 12 -out {out_image} -omat {matrix_out}"
    subprocess.run(flirt_cmd, shell=True)


def average_images(image_list, avg_image):
    """Average a list of images."""
    print("Averaging registered images...")
    add_cmd = ' -add '.join(image_list)
    fslmaths_cmd = f"fslmaths {add_cmd} -div {len(image_list)} {avg_image} -odt float"
    subprocess.run(fslmaths_cmd, shell=True)


def create_template(reference_image, image_set, iterations=1, n_jobs=-1):
    """Create a study-specific template image."""
    for iter_num in range(iterations):
        print(f"Starting iteration {iter_num + 1} of {iterations}...")

        # Parallelize the registration process
        Parallel(n_jobs=n_jobs)(delayed(run_flirt)(
            img,
            reference_image,
            os.path.join(os.path.dirname(img), f"registered_{iter_num}.nii.gz"),  # Save in the patient directory
            os.path.join(os.path.dirname(img), f"matrix_{iter_num}.mat")
        ) for img in image_set)

        # Update the list of registered images for averaging
        registered_images = [os.path.join(os.path.dirname(img), f"registered_{iter_num}.nii.gz") for img in image_set]

        # Average the images
        avg_image_path = os.path.join(directory, f"avg_im_iter_{iter_num}.nii.gz")
        average_images(registered_images, avg_image_path)

        # Update the reference image for the next iteration
        reference_image = avg_image_path

        print(f"Iteration {iter_num + 1} complete. Average image saved as {avg_image_path}")

    return avg_image_path

# Example usage
directory = "/home/user/NITRC_template_dataset"
reference_img = "/home/user/NITRC_template_dataset/BH0040/001.nii.gz"  # select a reference image without artifacts and a desired voxel spacing.

image_set = [os.path.join(directory, patient, "001.nii.gz") for patient in os.listdir(directory)]  # List of N images
iterations = 3  # Number of iterations for refinement of custom template

template_image = create_template(reference_img, image_set, iterations)
print(f"Template image created: {template_image}")
