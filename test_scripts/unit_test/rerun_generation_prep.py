# 遍历 generation tasks 文件夹下的所有子文件夹
# 删除子文件夹中的所有best_ckpt_generated_content文件夹，置空best_validation_result.json和experiment_results.csv
# 再遍历子文件夹的子子文件夹，删除generated_contents，置空result.json

import os
import shutil

def clear_file(file_path):
    """ 清空指定的文件内容，但对 JSON 和 CSV 文件进行特殊处理 """
    if file_path.endswith('.json'):
        # 对于 JSON 文件，写入空的 JSON 对象
        with open(file_path, 'w') as file:
            file.write('{}')
    elif file_path.endswith('.csv'):
        # 对于 CSV 文件，假设保留文件头（第一行）
        with open(file_path, 'r') as file:
            header = file.readline()
        with open(file_path, 'w') as file:
            file.write(header)
    else:
        # 对于其他文件类型，直接清空
        with open(file_path, 'w') as file:
            file.truncate(0)

def delete_folder(folder_path):
    """ 删除指定的文件夹，如果存在 """
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

def process_folder(folder_path):
    """ 处理每个子文件夹 """
    # 删除 best_ckpt_generated_content 文件夹
    delete_folder(os.path.join(folder_path, 'best_ckpt_generated_content'))

    # 置空 best_validation_result.json 和 experiment_results.csv 文件
    clear_file(os.path.join(folder_path, 'best_validation_result.json'))
    clear_file(os.path.join(folder_path, 'experiment_results.csv'))

    # 遍历子文件夹
    for sub_folder_name in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, sub_folder_name)
        if os.path.isdir(sub_folder_path):
            # 删除 generated_contents 文件夹
            delete_folder(os.path.join(sub_folder_path, 'generated_contents'))
            
            # 置空 result.json 文件
            clear_file(os.path.join(sub_folder_path, 'result.json'))

# 主目录
main_dir = '/home/cyzhao/generation_tasks_test'

# 遍历主目录下的所有子文件夹
for folder_name in os.listdir(main_dir):
    folder_path = os.path.join(main_dir, folder_name)
    if os.path.isdir(folder_path):
        process_folder(folder_path)
