import os
import json
from tqdm import tqdm

dirs_to_combine = [
    "/home/csgrad/sahmed9/reps/RealCQA/code/output/google/matcha-chartqa",
    "/home/csgrad/sahmed9/reps/RealCQA/code/outputs/google/matcha-chartqa", 
    "/home/csgrad/sahmed9/reps/RealCQA/code/outputsisi/google/matcha-chartqa"
]

output_dir = './output_matcha_chartqa'

#############

dirs_to_combine = [
    "/home/csgrad/sahmed9/reps/RealCQA/code/outputsisi/google/matcha-plotqa-v1"
]

output_dir = '/home/csgrad/sahmed9/reps/RealCQA/output_matcha_plotqav1'

###########

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store combined data
combined_data = {}

# Set to store all unique qa_id values
all_unique_qa_ids = set()

# Iterate through the specified directories
for directory in dirs_to_combine:
    for filename in tqdm(os.listdir(directory), desc=f"Processing files in {directory}"):
        # Check if the file is of the correct format
        if filename.startswith("__") and filename.endswith(".json"):
            version = filename[2]
            base_filename = filename[5:]

            # Load JSON data from the file
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)

            # Combine data into the dictionary using base filename without version
            if base_filename not in combined_data:
                combined_data[base_filename] = data
            else:
                # Extend data only with unique qa_id values
                existing_qa_ids = {entry['qa_id'] for entry in combined_data[base_filename]}
                new_entries = [entry for entry in data if entry['qa_id'] not in existing_qa_ids]
                combined_data[base_filename].extend(new_entries)

                # Update the set of all unique qa_id values
                all_unique_qa_ids.update(entry['qa_id'] for entry in data)

# Write combined data to new files
for base_filename, data in combined_data.items():
    output_filename = os.path.join(output_dir, base_filename)
    with open(output_filename, 'w') as output_file:
        json.dump(data, output_file, indent=2)

# Save the list of all unique qa_id values to a text file
unique_qa_ids_file = os.path.join(output_dir, 'unique_qa_ids.txt')
with open(unique_qa_ids_file, 'w') as ids_file:
    ids_file.write("\n".join(all_unique_qa_ids))

print("Combining and renaming completed.")
print("Total unique cobined : : ", len(all_unique_qa_ids) )