from datasets import load_from_disk
import os
import json
import re
import shutil
from functools import partial

PROMPT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.

USER:

{task_instruction}

ASSISTANT:

Okay.

{examples}

USER:
"""


def dataset_to_txt(datasst_path, file_path, file_name):
    # Load the dataset
    dataset = load_from_disk(datasst_path)

    # Open a new text file for writing
    new_file_name = file_path + '/' + file_name + '_with_multiround_prompts' + '.txt'
    with open(new_file_name, 'w') as file:
        count = 0
        for entry in dataset:
            # Extract the required fields from each entry
            model_input = entry['model_input']
            model_output = entry['model_output']
            ground_truth = entry['groud_truth']

            # Write the formatted data to the text file
            file.write(f"\n{count}:\n\n")
            file.write("------------------------------------------------\n")
            file.write(f"\n[INPUT]\n\n{model_input}\n")
            file.write(f"\n[OUPUT]\n\n{model_output}\n")
            file.write(f"\n[GROUND_TRUTH]\n\n{ground_truth}\n\n")
            file.write("------------------------------------------------\n")
            count += 1

    new_file_name = file_path + '/' + file_name + '.txt'
    with open(new_file_name, 'w') as file:
        count = 0
        for entry in dataset:
            # Extract the required fields from each entry
            raw_input = entry['model_input']
            last_user_index = raw_input.rfind("USER:")
            assistant_index = raw_input.find("ASSISTANT:", last_user_index)
            user_prefix_length = len("USER:\n")
            model_input = raw_input[last_user_index + user_prefix_length:assistant_index].strip()
            model_output = entry['model_output']
            ground_truth = entry['groud_truth']

            # Write the formatted data to the text file
            file.write(f"\n{count}:\n\n")
            file.write("------------------------------------------------\n")
            file.write(f"\n[INPUT]\n\n{model_input}\n")
            file.write(f"\n[OUPUT]\n\n{model_output}\n")
            file.write(f"\n[GROUND_TRUTH]\n\n{ground_truth}\n\n")
            file.write("------------------------------------------------\n")
            count += 1
    


def process_subfolders(base_path, target_base_path):

    subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]

    for subpath in subfolders:
        parts = subpath.split('/')[-1].split('_')
        experiment_name = parts[1]

        new_path = os.path.join(target_base_path, experiment_name)
        os.makedirs(new_path, exist_ok=True)
        
        file_path = subpath + '/best_validation_result.json'

        with open(file_path, 'r') as file:
            # 1. copy original generated dataset
            data = json.load(file)
            evaluate_result_path = data['evaluate_result_path']

            # Split the path and remove the last element
            split_path = evaluate_result_path.split('/')[:-1]
            trimmed_path = '/'.join(split_path)

            dataset_path = os.path.join(trimmed_path, 'dataset.txt')
            target_path = os.path.join(new_path, 'generated_dataset.txt')
            shutil.copy(dataset_path, target_path)

            # 2. add prompts into generated dataset inputs
            with open(dataset_path, 'r') as file:
                content = file.read()

            instruction = None
            examples = None
            json_path = "/home/cyzhao/main/NI_tasks/tasks.json"

            with open(json_path, "r", encoding="utf-8") as json_file:
                all_tasks = json.load(json_file)

                for task in all_tasks:
                    if task["task_name"] == experiment_name:
                        instruction = task["task_instruction"]
                        examples = task["examples"]
                        matches = re.findall(
                            r'\[input\]="(.*?)"\s*\[output\]="(.*?)"',
                            examples,
                            re.DOTALL,
                        )
                        assert matches != []
                        annotation_prompt_string = ""
                        for input, output in matches:
                            annotation_prompt_string += f"USER:\n\n{input}\n\n"
                            annotation_prompt_string += f"ASSISTANT:\n\n{output}\n\n"
                        assert annotation_prompt_string != ""

                        # Define the strings to be inserted
                        prefix = PROMPT_TEMPLATE.format(
                                task_instruction=instruction,
                                examples = annotation_prompt_string.strip()
                            )
                        suffix = """ASSISTANT:

"""
                        # Perform the replacements
                        modified_content = content.replace("[INPUT]\n\n", "[INPUT]\n\n" + prefix + "\n").replace("[OUPUT]\n\n", suffix + "\n[OUPUT]\n\n")

                        # Write the modified content back to the file or a new file
                        with open(os.path.join(new_path, 'generated_dataset_with_multiround_prompts.txt'), 'w') as file:
                            file.write(modified_content)

            best_ckpt_generated_content = os.path.join(subpath, 'best_ckpt_generated_content')
            dataset_to_txt(best_ckpt_generated_content, new_path, 'prediction') 

# Example usage
base_path = '/home/cyzhao/generation_tasks_best_3'
target_base_path = '/home/cyzhao/generation_tasks_best_ckpt_organized_data'
process_subfolders(base_path, target_base_path)