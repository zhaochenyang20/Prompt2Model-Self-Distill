import json
from datasets import Dataset


with open('task937_defeasible_nli_social_classification.json', 'r') as json_file:
    data = json.load(json_file)

instances = data['Instances'] 

print(len(instances))

test_dataset = instances[:3000]
eval_dataset = instances[3000:6000]
task_name = test_dataset[0]['id'].split('-')[0]

test_data_dict = {
    "input_col": [item["input"] for item in test_dataset],
    "output_col": [item["output"][0] for item in test_dataset],
}

eval_data_dict = {
    "input_col": [item["input"] for item in eval_dataset],
    "output_col": [item["output"][0] for item in eval_dataset],
}

test_dataset = Dataset.from_dict(test_data_dict)
print(test_dataset[0])  
test_dataset.save_to_disk(f"/home/cyzhao/prompt2model_test/testdataset/NI/test/{task_name}")
loaded_dataset = Dataset.load_from_disk(f"/home/cyzhao/prompt2model_test/testdataset/NI/test/{task_name}")
print(loaded_dataset[0])
print(loaded_dataset)

eval_dataset = Dataset.from_dict(eval_data_dict)
print(eval_dataset[0])
eval_dataset.save_to_disk(f"/home/cyzhao/prompt2model_test/testdataset/NI/eval/{task_name}")
loaded_dataset = Dataset.load_from_disk(f"/home/cyzhao/prompt2model_test/testdataset/NI/eval/{task_name}")
print(loaded_dataset[0])
print(loaded_dataset)