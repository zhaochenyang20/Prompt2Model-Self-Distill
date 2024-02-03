import os
import json
import shutil
from utils.path import ROOT

def copy_folder(source_folder_path, destination_folder_path):
    try:
        shutil.copytree(source_folder_path, destination_folder_path)
        print(f"文件夹成功复制从 {source_folder_path} 到 {destination_folder_path}")
    except Exception as e:
        print(f"复制文件夹时出错: {e}")

root_path = ROOT+'/generation_tasks_test'
best_eval_score = {}
best_test_score = {}
best_index = {}
generation_tasks = ["task034", "task035", "task036", "task039", "task121", "task281", "task569", "task671", "task1195","task1345", "task1557", "task1562", "task1622", "task1631", "task1659"]
for task in generation_tasks:
    best_eval_score[task] = 0
    best_index[task]= 0
    best_test_score[task]= 0


for sub_folder in os.listdir(root_path):
    if sub_folder.startswith('NI_task') and '_exp_' in sub_folder:
        # 从文件夹的名字中提取task和exp信息
        _, task, _, exp_number = sub_folder.split('_')

        # 读取best_validation_result.json文件
        result_file_path = os.path.join(root_path, sub_folder, 'best_validation_result.json')
        
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as result_file:
                result_data = json.load(result_file)
                
                # 获取validation_result的值
                validation_result = result_data.get('validation_result', 0)

                if task in best_eval_score:
                    old = best_eval_score[task]
                    best_eval_score[task] = max(best_eval_score[task], validation_result)
                    if old != best_eval_score[task]:
                        best_index[task] = exp_number
                        best_test_score[task] = result_data.get('test_result', 0)

destination_path_root = ROOT+'/generation_tasks_best_3'
for task in best_index.keys():
    source_path = root_path+ '/NI_' + task +'_exp_' + best_index[task]
    destination_path = destination_path_root + '/NI_' + task +'_exp_' + best_index[task]
    copy_folder(source_path, destination_path)