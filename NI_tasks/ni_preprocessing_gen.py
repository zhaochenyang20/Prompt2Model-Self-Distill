import json
from datasets import Dataset
from prompt2model.utils.path import TEST_DATA_ROOT, ROOT
import random


# TODO change json file path
folder_path = ROOT + "/main/NI_tasks/task_json/processed/"
files = [
    'task121_zest_text_modification.json',
    'task039_qasc_find_overlapping_words.json',
    'task036_qasc_topic_word_to_generate_related_fact.json',
    'task281_points_of_correspondence.json',
    'task1195_disflqa_disfluent_to_fluent_conversion.json',
    'task1345_glue_qqp_question_paraprashing.json',
    'task1562_zest_text_modification.json',
    'task1622_disfl_qa_text_modication.json'
]

for file_name in files:
    task_name = file_name.split('_')[0]
    file_name = folder_path + file_name
    with open(file_name) as json_file:
        data = json.load(json_file)

    instances = data["Instances"]

    eval_dataset, test_dataset = [], []
    random.shuffle(instances)  # 打乱每组数据以确保随机分配
    split_point = int(len(instances) * 0.1)  # 计算分割点，1:9 的比例
    eval_dataset.extend(instances[:split_point])  # 10% 的数据分到第一个数据集
    test_dataset.extend(instances[split_point:])  # 剩余 90% 分到第二个数据集
        


    test_data_dict = {
        "input_col": [item["input"] for item in test_dataset],
        "output_col": [item["output"][0] for item in test_dataset],
    }
    print(f"len of test: {len(test_data_dict)}")

    eval_data_dict = {
        "input_col": [item["input"] for item in eval_dataset],
        "output_col": [item["output"][0] for item in eval_dataset],
    }
    print(f"len of eval: {len(eval_data_dict)}")

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