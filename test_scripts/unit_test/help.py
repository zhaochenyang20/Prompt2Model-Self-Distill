import os
import json
import shutil

def copy_folder(source_folder_path, destination_folder_path):
    try:
        # 复制文件夹
        shutil.copytree(source_folder_path, destination_folder_path)
        print(f"文件夹成功复制从 {source_folder_path} 到 {destination_folder_path}")
    except Exception as e:
        print(f"复制文件夹时出错: {e}")

root_path = '/home/cyzhao/generation_tasks'
best_eval_score = {}
best_test_score = {}
best_index = {}

generation_tasks = ["task039", "task281", "task121", "task1195", "task034", "task1622", "task1562", "task671", "task1345", "task035", "task1659", "task569","task1631", "task1557", "task036"]
for task in generation_tasks:
    best_eval_score[task] = {20: 0, 40: 0}
    best_index[task]= {20: 0, 40: 0}
    best_test_score[task]= {20: 0, 40: 0}
experiments_key = {20: [1, 2, 3], 40: [4, 5, 6]}


for sub_folder in os.listdir(root_path):
    if sub_folder.startswith('NI_task') and '_exp_' in sub_folder:
        # 从文件夹的名字中提取task和exp信息
        _, task, _, exp_number = sub_folder.split('_')

        # 确定exp_number对应的字典experiments_key的key
        exp_key = 20 if int(exp_number) in experiments_key[20] else 40

        # 读取best_validation_result.json文件
        result_file_path = os.path.join(root_path, sub_folder, 'best_validation_result.json')
        
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as result_file:
                result_data = json.load(result_file)
                
                # 获取validation_result的值
                validation_result = result_data.get('validation_result', 0)

                if task in best_eval_score and exp_key in best_eval_score[task]:
                    old = best_eval_score[task][exp_key]
                    best_eval_score[task][exp_key] = max(best_eval_score[task][exp_key], validation_result)
                    if old != best_eval_score[task][exp_key]:
                        best_index[task][exp_key] = exp_number
                        best_test_score[task][exp_key] = result_data.get('test_result', 0)

destination_path_root = '/home/cyzhao/genetration_tasks_best'
for task in best_index.keys():
    for generation_batch_size in best_index[task].keys():
        source_path = root_path+ '/NI_' + task +'_exp_' + best_index[task][generation_batch_size]
        destination_path = destination_path_root + '/NI_' + task +'_exp_' + best_index[task][generation_batch_size]
        copy_folder(source_path, destination_path)