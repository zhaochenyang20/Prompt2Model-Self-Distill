import json
import os

task_json_file_path = "/home/cyzhao/main/NI_tasks/tasks.json"
folder_path = '/home/cyzhao/main/NI_tasks/task_json/unprocessed'

with open(task_json_file_path, 'r', encoding='utf-8') as file:
    existing_data = json.load(file)

if os.path.exists(folder_path):
    for file_path in os.listdir(folder_path):
        task_file_path = os.path.join(folder_path, file_path)
        if os.path.exists(task_file_path):
            print(task_file_path)
            with open(task_file_path, 'r', encoding='utf-8') as existing_file:
                task_data = json.load(existing_file)
                examples = task_data["Positive Examples"]
                example_string = ''
                for example in examples:
                    example_string +=  '[input]=' + example['input'] + '\n'
                    example_string +=  '[output]=' + example['output'] + '\n\n'

                new_data = {
                    "task_instruction": task_data["Definition"][0],
                    "task_name": task_data['Instances'][0]['id'].split('-')[0],
                    "examples": example_string,
                    "expected_content": ""
                }
                existing_data.append(new_data)
with open(task_json_file_path, 'w', encoding='utf-8') as updated_file:
    json.dump(existing_data, updated_file, ensure_ascii=False, indent=4)

# with open(task_json_file_path, 'a', encoding='utf-8') as updated_file:
#     for task in new_data:
#         json.dump(task, updated_file, ensure_ascii=False, indent=4)
#         updated_file.write(",\n")

print("提取和插入完成")

# 将提取的数据保存到新的 JSON 文件
# with open('tasks.json', 'w', encoding='utf-8') as new_file:
#     json.dump(new_data, new_file, ensure_ascii=False, indent=4)

# print("提取和保存完成")
