# Registration to a custom TOF MRA template
import os
import subprocess
import nibabel as nib
import numpy
import numpy as np
from scipy import ndimage
import torchio as tio
from joblib import Parallel, delayed

def custom_crop_and_pad(input_path, output_path, target_size=(128, 128, 64), crop_bounds=(None, None, None)):

    # Load the image using nibabel
    nib_image = nib.load(input_path)
    image_data = nib_image.get_fdata()
    original_affine = nib_image.affine

    current_size = image_data.shape

    # Apply initial cropping if specified
    for dim, bounds in enumerate(crop_bounds):
        if bounds is not None:
            start, end = bounds
            # Ensure the bounds are within the current size
            end = min(end, current_size[dim])
            image_data = np.take(image_data, indices=range(start, end), axis=dim)

    # Convert the cropped image to torchio ScalarImage
    cropped_image = tio.ScalarImage(tensor=image_data[np.newaxis, ...], affine=original_affine)  # Add channel dimension

    # Pad the image to target size using torchio
    transform = tio.CropOrPad(target_size)
    transformed_image = transform(cropped_image)

    # Save the transformed image
    transformed_image.save(output_path)


def SYNTHSTRIP(input, output, output_mask, use_GPU=True ):
    if use_GPU:
        command = f"mri_synthstrip -i {input} -o {output} -m {output_mask} --gpu"
    else:
        command = f"mri_synthstrip -i {input} -o {output} -m {output_mask}"

    subprocess.run(command, shell=True)
    assert os.path.exists(output)

def reorient_to_std(input_image_path, output_image_path=None):

    if output_image_path is None:
        output_image_path = input_image_path


    # Construct and run the command
    command = f"fslreorient2std {input_image_path} {output_image_path}"
    print(command)
    subprocess.run(command, shell=True, check=True)

def skullstrip_patient(patient, dataset):
    print("Processing patient: ", patient)
    patient_folder = os.path.join(dataset, patient)
    tof_image = os.path.join(patient_folder, "001.nii.gz")
    print(tof_image)

    tof_image_BET = os.path.join(patient_folder, "001_BET.nii.gz")

    tof_image_BET_mask = os.path.join(patient_folder, "001_BET_mask.nii.gz")


    tof_image_registered_output_BET = tof_image_BET.replace(".nii.gz", "_registered.nii.gz")
    tof_image_registered_output = tof_image.replace(".nii.gz", "_registered.nii.gz")


    output_matrix_path = os.path.join(patient_folder, "reg_matrix_TOF_to_MNI.mat")


    # assign new variables to TOF for better readability
    TOF = tof_image_registered_output

    TOF_cropped = TOF.replace(".nii.gz",
                                          "_cropped_" + str(target_size[0]) + "_"
                                          + str(target_size[1])
                                          + "_" + str(target_size[2])+ ".nii.gz")

    # Reorient to standard MNI orientation
    reorient_to_std(tof_image)

    # Skullstripping with SYNTHSTRIP
    SYNTHSTRIP(tof_image, tof_image_BET, tof_image_BET_mask, use_GPU=False)

    assert os.path.exists(tof_image_BET)
    print("Skullstripping DONE ", patient)

def process_patient(patient, dataset, perform_registration=False, connected_component=False):
    print("Processing patient: ", patient)
    patient_folder = os.path.join(dataset, patient)
    tof_image = os.path.join(patient_folder, "001.nii.gz")

    tof_image_BET = os.path.join(patient_folder, "001_BET.nii.gz")
    tof_image_BET_mask = os.path.join(patient_folder, "001_BET_mask.nii.gz")


    tof_image_registered_output_BET = tof_image_BET.replace(".nii.gz", "_registered.nii.gz")
    tof_image_registered_output = tof_image.replace(".nii.gz", "_registered.nii.gz")

    output_matrix_path = os.path.join(patient_folder, "reg_matrix_TOF_to_template.mat")

    # assign new variables to TOF for better readability
    TOF = tof_image_registered_output

    TOF_cropped = TOF.replace(".nii.gz",
                                          "_cropped_" + str(target_size[0]) + "_"
                                          + str(target_size[1])
                                          + "_" + str(target_size[2])+ ".nii.gz")

    if perform_registration:
        # Reorient to standard MNI orientation
        reorient_to_std(tof_image)

        # Skullstripping with SYNTSTRIP
        SYNTHSTRIP(tof_image, tof_image_BET, tof_image_BET_mask, use_GPU=False)

        assert os.path.exists(tof_image_BET)

        # Registration
        command = (f"flirt -in {tof_image_BET} -ref {TOF_custom_template} -out {tof_image_registered_output_BET} "
                   f"-omat {output_matrix_path} "
                   f"-searchcost {cost_function_registration} "
                   f"-cost {cost_function_registration} -dof {dof}")

        subprocess.run(command, shell=True, check=True)

        # Apply transformation matrix resulting from BET to custom TOF template to TOF
        command_apply_xfm_TOF = (f"flirt -in {tof_image} -ref {TOF_custom_template} "
                             f"-out {tof_image_registered_output} -init {output_matrix_path} -applyxfm ")


        subprocess.run(command_apply_xfm_TOF, shell=True, check=True)

    # Crop or pad an image based on custom slicing
    custom_crop_and_pad(TOF, TOF_cropped, target_size=target_size,
                        crop_bounds=cropping_bound_slices)

# define path of your datasets in a list
roots = ["/home/user/IXI_dataset"]  # dataset containing one folder for each patient, each folder contains a nii.gz image.

# Define freesurfer location
freesurfer_home = '/home/user/freesurfer'  # Replace with your FreeSurfer installation path
os.environ['FREESURFER_HOME'] = freesurfer_home
os.environ['PATH'] += os.pathsep + os.path.join(freesurfer_home, 'bin')

# custom TOF MRA template
TOF_custom_template = "./sample_data/TOF_template_NITRC_BET.nii.gz"  # input the path to the custom TOF MRA template provided in this repository.

# PARAMETERS FOR PREPROCESSING
# target size
target_size = (128, 128, 32)  # the dimensions are swapped to (32,128,128) during training (N, Channel, D, H, W)
cropping_bound_slices = [(80, 208), (120, 248), (40, 72)]  # set cropping boundaries on the template image.

# Registration
cost_function_registration = "mutualinfo"
dof = 6  # degrees of freedom

# if only cropping is necessary set both to False, if preprocessing a dataset for the first time set to True.
perform_registration = True
perform_skull_stripping = True

for dataset in roots:
    patients = os.listdir(dataset)
    print(patients)
    if perform_skull_stripping:
        Parallel(n_jobs=4)(delayed(skullstrip_patient)(patient, dataset) for patient in patients)
    Parallel(n_jobs=-1)(delayed(process_patient)(patient, dataset, perform_registration, connected_component=False) for patient in patients)

