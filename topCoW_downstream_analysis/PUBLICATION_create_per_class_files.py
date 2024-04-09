import nibabel as nib
import os
import numpy as np
import re

# This script extracts individual artery segments from a single nii.gz file containing all multiclass segmentations.
# This is a preliminary step before running the EvaluateSegmentation tool
# https://github.com/Visceral-Project/EvaluateSegmentation

# Folder containing segmentation results in nii.gz format.
input_segmentation_multiclass_folder = "/home/user/nnUNet/segmentatation_CoW"
folder_name = os.path.basename(input_segmentation_multiclass_folder)

output_dir = "/home/user/segmentation_evaluation"

def extract_patient_id(filename):
    match = re.search(r'COW_(\d+)\.nii\.gz', filename)# correction from SAH to BRAIN to fit the new project. COW_
    if match:
        return match.group(1)
    return None


patients = []
for file in os.listdir(input_segmentation_multiclass_folder):
    print(file)
    print(extract_patient_id(file))
    patient_id = extract_patient_id(file)
    if ".nii.gz" in file:
        patients.append(str(patient_id))

print(patients)
def extract_class_label_from_segmentation(input_dir, patients, num_classes, out_name_tag):
    for patient in patients:
        if file.endswith(".nii.gz"):
            input_path = os.path.join(input_dir,  "COW_" + str(patient) + ".nii.gz")
            nifti_orig = nib.load(input_path)
            print(' - nifti loaded from:', input_path)
            print(' - dimensions of the loaded nifti: ', nifti_orig.shape)
            print(' - nifti data type:', nifti_orig.get_data_dtype())
            label_mat = nifti_orig.get_fdata()
            for i in range(1, num_classes):
                binary_class_label_array = np.zeros(label_mat.shape)
                binary_class_label_array[label_mat == i] = 1
                binary_class_nifti_img = nib.Nifti1Image(binary_class_label_array, nifti_orig.affine)

                os.makedirs(os.path.join(output_dir, folder_name,  str(i)), exist_ok=True)

                output_path = os.path.join(output_dir, folder_name, str(i), out_name_tag + "_" + str(patient) + ".nii.gz")
                nib.save(binary_class_nifti_img, output_path)

extract_class_label_from_segmentation(input_segmentation_multiclass_folder,
                                      patients,
                                      num_classes=8,
                                      out_name_tag="COW")


