import json
import os
from collections import defaultdict

# Replace 'path/to/your/directory' with the path to your directory containing JSON files
directory_path = './output'

# Initialize a defaultdict to store grouped data
grouped_data = defaultdict(lambda: {'accuracies': [], 'solution_times': []})
file_count = 0
# Loop through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('results.json'):
        file_path = os.path.join(directory_path, filename)
        file_count += 1
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Grouping data by (jaro_th, jaccard_th, minhash_signature_size)
        for key, value in data.items():
            group_key = (value['jaro_th'], value['jaccard_th'], value['minhash_signature_size'])
            grouped_data[group_key]['accuracies'].append(value['accuracy'])
            grouped_data[group_key]['solution_times'].append(value['solution_time'])

# Calculating mean accuracy and solution time for each group
mean_results = {}

for group_key, group_values in grouped_data.items():
    mean_accuracy = sum(group_values['accuracies']) / len(group_values['accuracies'])
    mean_solution_time = sum(group_values['solution_times']) / len(group_values['solution_times'])
    mean_results[group_key] = {
        'mean_accuracy': mean_accuracy,
        'mean_solution_time': mean_solution_time
    }

print(file_count)
# Printing the results
for group_key, means in mean_results.items():
    print(f"Group {group_key}:")
    print(f"  Mean Accuracy: {means['mean_accuracy']}")
    print(f"  Mean Solution Time: {means['mean_solution_time']}\n")
