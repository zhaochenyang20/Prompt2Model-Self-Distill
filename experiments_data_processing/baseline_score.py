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
    # if model_output == groud_truth:
    #     return 1
    # else:
    #     return 0
    
    if model_output!= '':
        return int(model_output in groud_truth or groud_truth in model_output)
    else:
        return 0

for sub_folder in os.listdir(folder_path):
    prefix, task_type, task_name = sub_folder.split('_')
    if prefix == '20240310':
        genre = 'classification'
        if task_name not in classification_tasks:
            genre = 'generation'
            continue
        print(sub_folder)
        dataset = load_from_disk(f'{folder_path}/{sub_folder}')
        df = dataset.to_pandas()
        df['correct'] = df.apply(check_correct, axis=1)
        correct_sum = df['correct'].sum()
        total_rows = len(df)
        accuracy = correct_sum / total_rows

        print(f'{task_type}:{accuracy}')
        if task_name not in res[genre]:
            res[genre][task_name] = []
        res[genre][task_name].append(
            (task_type, accuracy)
        )

print(res)

# {'classification': {'task284': [('test', 0.05677154582763338), ('eval', 0.05238828967642527)], 'task1612': [('test', 0.5117283950617284), ('eval', 0.5166666666666667)], 'task199': [('test', 0.27384615384615385), ('eval', 0.2711864406779661)], 'task1385': [('eval', 0.30612244897959184), ('test', 0.3307174887892377)], 'task1529': [('test', 0.029637760702524697), ('eval', 0.0297029702970297)], 'task738': [('eval', 0.5703125), ('test', 0.5872244402013539)], 'task346': [('eval', 0.3436055469953775), ('test', 0.3348153214774282)], 'task200': [('eval', 0.44719101123595506), ('test', 0.4508584224931575)], 'task1386': [('test', 0.3329608938547486), ('eval', 0.32653061224489793)], 'task329': [('eval', 0.25733634311512416), ('test', 0.23964053919121317)], 'task937': [('eval', 0.37596302003081666), ('test', 0.368034188034188)], 'task1615': [('eval', 0.0056179775280898875), ('test', 0.005558987029030266)], 'task1516': [('eval', 0.0), ('test', 0.0)], 'task190': [('eval', 0.1325115562403698), ('test', 0.1459579559049735)]}, 'generation': {}}
# {'classification': {'task284': [('test', 0.9006497948016415), ('eval', 0.8967642526964561)], 'task1612': [('test', 0.5135802469135803), ('eval', 0.5166666666666667)], 'task199': [('test', 0.3162393162393162), ('eval', 0.31741140215716485)], 'task1385': [('eval', 0.30612244897959184), ('test', 0.33183856502242154)], 'task1529': [('test', 0.08562019758507135), ('eval', 0.09306930693069307)], 'task738': [('eval', 0.5828125), ('test', 0.5964242319041833)], 'task346': [('eval', 0.3682588597842835), ('test', 0.35174418604651164)], 'task200': [('eval', 0.46741573033707867), ('test', 0.46752923612839015)], 'task1386': [('test', 0.3396648044692737), ('eval', 0.32653061224489793)], 'task329': [('eval', 0.30248306997742663), ('test', 0.291063404892661)], 'task937': [('eval', 0.4591679506933744), ('test', 0.4711111111111111)], 'task1615': [('eval', 0.0056179775280898875), ('test', 0.005558987029030266)], 'task1516': [('eval', 0.15942028985507245), ('test', 0.1765625)], 'task190': [('eval', 0.28043143297380585), ('test', 0.2700393095197402)]}, 'generation': {}}