import os
import xml.etree.cElementTree as ET
import csv
import pandas as pd
import re

# This script evaluates the segmentation results against real ground truht labels using:
# https://github.com/Visceral-Project/EvaluateSegmentation

def parse_xml_to_csv_new(xml_path, csv_path):
    # Parse XML and extract metrics into a dictionary
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data_dict = {child.attrib["symbol"]: child.attrib["value"] for child in root.findall(".//metrics/*")}

    if not data_dict:
        data_dict = {key: None for key in metrics_list}

    df = pd.DataFrame([data_dict])
    df.to_csv(csv_path, index=False)
    os.remove(xml_path)


def segment_comparison(goldstandard_path, segmentation_path, executable_path, eval_result_path, threshold, measures):
    print(measures)
    command_string = executable_path + " \"" + goldstandard_path + "\" \"" + segmentation_path + "\" -use " + measures + " -xml \"" + eval_result_path + "\" -thd " + str(
        threshold) + " -unit voxel"
    print(command_string)
    os.system(command_string)


def extract_patient_id(filename):
    match = re.search(r'COW_(\d+)\.nii\.gz', filename)
    if match:
        return match.group(1)
    return None


def extract_patient_id_from_csv(filename):
    match = re.search(r'results_(\d+)_all_metrics\.csv', filename)
    if match:
        return match.group(1)
    return None


def combine_csvs(directory):
    # List to store dataframes
    dfs = []

    # Loop through each CSV in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Create the full file path
            file_path = os.path.join(directory, filename)
            print(file_path)

            df = pd.read_csv(file_path)

            # Add the patient_id column
            df['patient_id'] = extract_patient_id_from_csv(filename)

            dfs.append(df)

    # Concatenate all the dataframes into one
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.set_index("patient_id")

    # Write the combined dataframe to a new CSV
    combined_df.to_csv(os.path.join(summary_directory, 'dataset_results.csv'))

    return combined_df


root = "/home/user/segmentation_evaluation"

num_classes = 8
threshold = 0.5

segmentation_dir = os.path.join(root, "segmentations_by_student_model")
ground_truth_dir = os.path.join(root, "Labels_real")

measures = "DICE,HDRFDST@0.95@,bAVD,AVGDIST,TP,FP,TN,FN,REFVOL,SEGVOL"

metrics_list = ["DICE", "HDRFDST", "bAVD", "AVGDIST", "TP", "FP", "TN", "FN", "REFVOL", "SEGVOL"]  # evaluation metrics

results_folder = "EvaluateSegmentation_results"  # folder to store the results.

summary_folder = "summary_results"
results_dir = os.path.join(root, results_folder)  # the path of the results file

executable_path = os.path.join(root, "EvaluateSegmentation") # path where the EvaluateSegmentation executable is located.

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

for i in range(1, num_classes):

    class_number = i
    class_number = str(class_number)

    results_class_dir = os.path.join(results_dir, class_number)
    if not os.path.exists(results_class_dir):
        os.makedirs(results_class_dir)

    summary_directory = os.path.join(results_class_dir, summary_folder)

    if not os.path.exists(summary_directory):
        os.makedirs(summary_directory)

    segmentation_results_dir = os.path.join(segmentation_dir, class_number)  # folder where the model segmentations are stored.
    ground_truth_label_dir = os.path.join(ground_truth_dir, class_number)   # folder where the reference/ground_truth segmentations are stored


    for file in os.listdir(segmentation_results_dir):
        print(file)
        print(extract_patient_id(file))
        patient_id = extract_patient_id(file)
        if file.endswith(".nii.gz"):
            seg = os.path.join(segmentation_results_dir, "COW_" + str(patient_id) + ".nii.gz")
            label = os.path.join(ground_truth_label_dir, "COW_" + str(patient_id) + ".nii.gz")
            eval_results_save_xml_path = os.path.join(root, results_folder, class_number, "results_" + str(patient_id) + ".xml")
            segment_comparison(label, seg, executable_path, eval_results_save_xml_path, 0.5, measures)
            eval_results_save_csv_path = os.path.join(root, results_folder, class_number, "results_" + str(patient_id) + "_all_metrics.csv")
            print(eval_results_save_csv_path)
            parse_xml_to_csv_new(eval_results_save_xml_path, eval_results_save_csv_path)

    combined_df = combine_csvs(results_class_dir)
    print(combined_df.columns)

    df = combined_df
    # Calculate the desired statistics for each column
    mean_values = df.mean()
    median_values = df.median()
    std_values = df.std()
    variance_values = df.var()

    # Combine the results into a new dataframe
    stats_df = pd.DataFrame({
        'Metric': df.columns,
        'Mean': mean_values,
        'Median': median_values,
        'Standard Deviation': std_values,
        'Variance': variance_values
    })

    # Save the results
    stats_df.to_csv(os.path.join(summary_directory, 'summary_statistics.csv'), index=False)


file_paths = [os.path.join(root, f"EvaluateSegmentation_results/{i}/summary_results/summary_statistics.csv") for i in range (1, 8)]

print(file_paths)
classes = ['basilar', 'PCA', "ICA", 'M1', "ACA", "Pcom", "Acom"] # 7 classes in total

dataframes = [pd.read_csv(path) for path in file_paths]

# Add a 'Class' column to each dataframe
for dataframe, cls in zip(dataframes, classes):
    dataframe['Class'] = cls

# Concatenate all the dataframes into one
combined_df = pd.concat(dataframes)

combined_df.groupby('Class')

print(combined_df)

combined_df.to_csv(os.path.join(root, "EvaluateSegmentation_results/aggregated_dataframe.csv"))

