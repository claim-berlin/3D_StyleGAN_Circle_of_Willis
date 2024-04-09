import os
import nibabel as nib


# This script maps the provided labels by the TopCoW Challenge https://topcow23.grand-challenge.org/ using a dictionary.
# This was used to merge the left and right labels and the ACA class.

# Specify the directory containing the NIfTI files
nifti_dir = '/home/user/Labels'

# Mapping of old labels to new labels
label_mapping = {
    1: 1,  # Basilar
    2: 2,  # P1_L
    3: 2,  # P1_R
    4: 3,  # ICA_L
    5: 4,  # M1_L
    6: 3,  # ICA_R
    7: 4,  # M1_R
    8: 6,  # Pcom_L
    9: 6,  # Pcom_R
    10: 7, # Acom
    11: 5, # A1_L
    12: 5, # A1_R
    13: 5, # 3rd_A2
}

# Iterate over all files in the directory
for filename in os.listdir(nifti_dir):
    if filename.endswith('.nii') or filename.endswith('.nii.gz'):
        file_path = os.path.join(nifti_dir, filename)

        # Load the NIfTI file
        nifti = nib.load(file_path)
        data = nifti.get_fdata()

        # Apply label remapping
        for old_label, new_label in label_mapping.items():
            data[data == old_label] = new_label

        # Create a new NIfTI image with the modified data
        # This retains the original header and affine matrix
        new_nifti = nib.Nifti1Image(data, affine=nifti.affine, header=nifti.header)

        nib.save(new_nifti, file_path)

        print(f'Processed and overwritten: {file_path}')

print('All NIfTI files have been processed.')
