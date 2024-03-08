import json
from datasets import Dataset
from prompt2model.utils.path import TEST_DATA_ROOT, ROOT
from collections import Counter
from collections import defaultdict
import random


def get_highest_digit(number):
    if number == 0:
        return 0
    temp = number
    while temp >= 10:
        temp //= 10

    highest_digit = temp * 10 ** (len(str(number)) - 1)

    return highest_digit


# TODO change json file path
folder_path = ROOT + "/main/NI_tasks/task_json/processed/"
files = [
    'task199_mnli_classification.json',
    'task200_mnli_entailment_classification.json',
    'task738_perspectrum_classification.json',
    'task937_defeasible_nli_social_classification.json',
    'task1385_anli_r1_entailment.json',
    'task1386_anli_r2_entailment.json',
    'task1516_imppres_naturallanguageinference.json',
    'task1529_scitail1.1_classification.json',
    'task1612_sick_label_classification.json',
    'task1615_sick_tclassify_b_relation_a.json',
    'task190_snli_classification.json'
]

for file_name in files:
    task_name = file_name.split('_')[0]
    file_name = folder_path + file_name
    with open(file_name) as json_file:
        data = json.load(json_file)

    instances = data["Instances"]

    grouped_data = defaultdict(list)
    for instance in instances:
        output_type = tuple(instance['output'])  # 将output转换为元组，以便作为字典的键
        grouped_data[output_type].append(instance)
    
    eval_dataset, test_dataset = [], []
    for output, data in grouped_data.items():
        random.shuffle(data)  # 打乱每组数据以确保随机分配
        split_point = int(len(data) * 0.1)  # 计算分割点，1:9 的比例
        eval_dataset.extend(data[:split_point])  # 10% 的数据分到第一个数据集
        test_dataset.extend(data[split_point:])  # 剩余 90% 分到第二个数据集
        


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
        f"{TEST_DATA_ROOT}/prompt2model_test/testdataset/NI/test/{task_name}"
    )
    loaded_dataset = Dataset.load_from_disk(
        f"{TEST_DATA_ROOT}/prompt2model_test/testdataset/NI/test/{task_name}"
    )

    eval_dataset = Dataset.from_dict(eval_data_dict)
    eval_dataset.save_to_disk(
        f"{TEST_DATA_ROOT}/prompt2model_test/testdataset/NI/eval/{task_name}"
    )
    loaded_dataset = Dataset.load_from_disk(
        f"{TEST_DATA_ROOT}/prompt2model_test/testdataset/NI/eval/{task_name}"
    )