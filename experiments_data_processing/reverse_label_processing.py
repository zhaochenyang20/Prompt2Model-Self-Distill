from datasets import load_from_disk, Dataset
import numpy as np
from pathlib import Path

# TODO: 1/3 change classification task name
task = '937'
# TODO: 2/3 add dataset path
dataset_path = '/home/azureuser/p2mss/p2mss/NI_task937_exp_3/task937_0.7_True_False_3/dataset'
# TODO: 3/3 change task labels
labels = ['strengthener', 'weakener']

root = '/home/azureuser/p2mss/p2mss/reverse_label_data'
folder_path = Path(root) / task
dataset = load_from_disk(dataset_path)


def count_diff_output_col(dataset1: Dataset, dataset2: Dataset) -> int:
    """
    return number of data points with different 'output_col'
    
    args:
    - dataset1: original dataset
    - dataset2: flipped dataset
    
    return:
    - number of lines
    """

    if len(dataset1) != len(dataset2):
        raise ValueError("The two datasets must have the same number of rows.")
    
    mismatches = 0
    for idx in range(len(dataset1)):
        if dataset1[idx]['output_col'] != dataset2[idx]['output_col']:
            print(f"original dataset label:{dataset1[idx]['output_col']} => new dataset label:{dataset2[idx]['output_col']}")
            mismatches += 1

    return mismatches


# create a function to flip 'output_col' label
def make_flip_label_function(indices_to_flip):
    def flip_label(example, index):
        new_example = example.copy()
        if index in indices_to_flip:
            new_label = labels[0] if example['output_col'] == labels[1] else labels[1]
            new_example['output_col'] = new_label
        return new_example
    return flip_label

# flip ratio
ratios = [0.25, 0.5, 0.75, 1.0]

# go through every flip atio, and then keep the dataset
for ratio in ratios:
    # create random flip indices
    indices_to_flip = np.random.choice(range(len(dataset)), size=int(len(dataset)*ratio), replace=False)
    
    flip_label = make_flip_label_function(indices_to_flip)
    # create a flipped dataset
    flipped_dataset = dataset.map(flip_label, with_indices=True)
    
    # keep a flipped dataset
    save_path = folder_path / f"flipped_dataset_{int(ratio*100)}"
    flipped_dataset.save_to_disk(save_path)
    print(f"Dataset with {int(ratio*100)}% labels flipped saved to {save_path}.")

    mismatches_count = count_diff_output_col(dataset, flipped_dataset)
    print(f"Number of rows with different 'output_col': {mismatches_count}")

    
