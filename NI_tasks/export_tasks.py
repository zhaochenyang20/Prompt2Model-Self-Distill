import json
import os
import shutil
import sys

import requests
from datasets import Dataset
from utils.path import ROOT

sys.path.append(ROOT + "/main/test_scripts/unit_test")
# from test_ni import evaluate_model

# step0: specify the json files to be processed
# TODO paste name of json files
github_raw_urls = [
    "https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks/task642_esnli_classification.json"
]
unprocessed_file_folder = ROOT + "/main/NI_tasks/task_json/unprocessed/"
for github_raw_url in github_raw_urls:
    local_file_path = unprocessed_file_folder + github_raw_url.split("/")[-1]

    response = requests.get(github_raw_url)
    print(response)
    if response.status_code == 200:

        # 将文件内容保存到本地文件
        with open(local_file_path, "wb") as file:
            file.write(response.content)
        print(f"文件已下载到 {local_file_path}")
    else:
        print("无法下载文件，HTTP状态码：", response.status_code)


# step1: 导入task desciption到task json文件
task_json_file_path = ROOT + "/main/NI_tasks/tasks.json"
folder_path = ROOT + "/main/NI_tasks/task_json/unprocessed"
destination_folder = ROOT + "/main/NI_tasks/task_json/processed"

files = []

with open(task_json_file_path, "r", encoding="utf-8") as file:
    existing_data = json.load(file)

if os.path.exists(folder_path):
    for file_path in os.listdir(folder_path):
        task_file_path = os.path.join(folder_path, file_path)
        if os.path.exists(task_file_path):
            print(task_file_path)
            with open(task_file_path, "r", encoding="utf-8") as existing_file:
                task_data = json.load(existing_file)
                examples = task_data["Positive Examples"]
                example_string = ""
                for example in examples:
                    example_string += '[input]="' + example["input"] + '"\n'
                    example_string += '[output]="' + example["output"] + '"\n\n'

                new_data = {
                    "task_instruction": task_data["Definition"][0],
                    "task_name": task_data["Instances"][0]["id"].split("-")[0],
                    "examples": example_string,
                    "expected_content": "",
                    "optional_list": [],
                }
                existing_data.append(new_data)
                files.append(task_file_path)

with open(task_json_file_path, "w", encoding="utf-8") as updated_file:
    json.dump(existing_data, updated_file, ensure_ascii=False, indent=4)
# 将提取的数据保存到新的 JSON 文件
# with open('tasks.json', 'w', encoding='utf-8') as new_file:
#     json.dump(new_data, new_file, ensure_ascii=False, indent=4)

# print("提取和保存完成")
print("提取和插入完成")

# step2: 分割出test和validation的数据集
def get_highest_digit(number):
    if number == 0:
        return 0
    temp = number
    while temp >= 10:
        temp //= 10

    highest_digit = temp * 10 ** (len(str(number)) - 1)

    return highest_digit


task_names = []

for file_name in files:
    with open(file_name) as json_file:
        data = json.load(json_file)

    instances = data["Instances"]

    if len(instances) < 2000:
        data_size = get_highest_digit(len(instances))
        test_dataset = instances[: data_size // 2]
        eval_dataset = instances[data_size // 2 : data_size]
    else:
        test_dataset = instances[:1000]
        eval_dataset = instances[1000:2000]
    task_name = test_dataset[0]["id"].split("-")[0]
    task_names.append(task_name)

    print(
        f"{task_name}: evaluation data size = {len(eval_dataset)}, test data size = {len(test_dataset)}"
    )

    test_data_dict = {
        "input_col": [item["input"] for item in test_dataset],
        "output_col": [item["output"][0] for item in test_dataset],
    }

    eval_data_dict = {
        "input_col": [item["input"] for item in eval_dataset],
        "output_col": [item["output"][0] for item in eval_dataset],
    }

    test_dataset = Dataset.from_dict(test_data_dict)
    test_dataset.save_to_disk(
        f"{ROOT}/prompt2model_test/testdataset/NI/test/{task_name}"
    )
    loaded_dataset = Dataset.load_from_disk(
        f"{ROOT}/prompt2model_test/testdataset/NI/test/{task_name}"
    )

    eval_dataset = Dataset.from_dict(eval_data_dict)
    eval_dataset.save_to_disk(
        f"{ROOT}/prompt2model_test/testdataset/NI/eval/{task_name}"
    )
    loaded_dataset = Dataset.load_from_disk(
        f"{ROOT}/prompt2model_test/testdataset/NI/eval/{task_name}"
    )

    shutil.move(file_name, destination_folder)

# step3: 测试
# evaluate_model(task_names)

# step4: 加expected content和metrics
