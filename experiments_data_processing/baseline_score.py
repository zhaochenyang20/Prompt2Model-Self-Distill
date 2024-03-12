from datasets import load_from_disk
import os

folder_path = '/home/azureuser/p2mss/p2mss/baseline_generated_data'

res = {
    'classification': {},
    'generation': {}
}

classification_tasks = ['task346', 'task190', 'task199', 'task1612', 'task200', 'task738', 'task937', 'task1385', 'task1386', 'task1516', 'task1529', 'task1615', 'task284', 'task329']

def check_correct(row):
    model_output = row['model_output']
    groud_truth = row['groud_truth']
    if model_output!= '':
        return int(model_output in groud_truth or  groud_truth in model_output)
    else:
        return 0

for sub_folder in os.listdir(folder_path):
    prefix, task_type, task_name = sub_folder.split('_')
    if prefix == '20240310':
        dataset = load_from_disk(f'{folder_path}/{sub_folder}')
        df = dataset.to_pandas()
        df['correct'] = df.apply(check_correct, axis=1)
        correct_sum = df['correct'].sum()
        total_rows = len(df)
        accuracy = correct_sum / total_rows
        genre = 'classification'
        if task_name not in classification_tasks:
            genre = 'generation'

        if task_name not in res:
            res[genre][task_name] = []
        res[genre][task_name].append(
            (task_type, accuracy)
        )

print(res)

