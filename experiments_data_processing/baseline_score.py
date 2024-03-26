import os
from datasets import load_from_disk
from collections import Counter
import sys

sys.path.append(os.path.abspath('/home/azureuser/p2mss/p2mss/main/test_scripts/utils'))
from evaluate import exact_match_score

folder_path = '/home/azureuser/p2mss/p2mss/baseline_generated_data'

res = {
    'classification': {},
    'generation': {}
}

classification_tasks = ['task346', 'task190', 'task199', 'task1612', 'task200', 'task738', 'task937', 'task1385', 'task1386', 'task1516', 'task1529', 'task1615', 'task284', 'task329']


for sub_folder in os.listdir(folder_path):
    if not sub_folder.startswith('20240310'):
        continue
    prefix, task_type, task_name = sub_folder.split('_')
    if prefix == '20240310':
        genre = 'classification'
        if task_name not in classification_tasks:
            genre = 'generation'
            continue
        if task_type == 'eval':
            continue
        dataset = load_from_disk(f'{folder_path}/{sub_folder}')
        df = dataset.to_pandas()
        labels = list(Counter(df['groud_truth']).keys())
        score = exact_match_score(df['groud_truth'], df['model_output'])
        total_rows = len(df)
        accuracy = score / total_rows

        if task_name not in res[genre]:
            res[genre][task_name] = []
        res[genre][task_name].append(
            (task_type, accuracy)
        )

print(res)

# == 
{'classification': {'task284': [('test', 0.05677154582763338)], 'task1612': [('test', 0.5117283950617284)], 'task199': [('test', 0.27384615384615385)], 'task1529': [('test', 0.029637760702524697)], 'task1385': [('test', 0.3307174887892377)], 'task346': [('test', 0.3348153214774282)], 'task1386': [('test', 0.3329608938547486)], 'task329': [('test', 0.23964053919121317)], 'task738': [('test', 0.5872244402013539)], 'task1516': [('test', 0.0)], 'task1615': [('test', 0.005558987029030266)], 'task190': [('test', 0.1459579559049735)], 'task937': [('test', 0.368034188034188)], 'task200': [('test', 0.4508584224931575)]}, 'generation': {}}
# double in
{'classification': {'task284': [('test', 0.9006497948016415)], 'task1612': [('test', 0.5135802469135803)], 'task199': [('test', 0.3162393162393162)], 'task1529': [('test', 0.08562019758507135)], 'task1385': [('test', 0.33183856502242154)], 'task346': [('test', 0.35174418604651164)], 'task1386': [('test', 0.3396648044692737)], 'task329': [('test', 0.291063404892661)], 'task738': [('test', 0.5964242319041833)], 'task1516': [('test', 0.1765625)], 'task1615': [('test', 0.005558987029030266)], 'task190': [('test', 0.2700393095197402)], 'task937': [('test', 0.4711111111111111)], 'task200': [('test', 0.46752923612839015)]}, 'generation': {}}
# new metric
