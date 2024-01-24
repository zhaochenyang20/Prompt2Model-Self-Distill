import os
import pandas as pd

class ExperimentType:
    def __init__(self, root_path, total_experiments):
        self.root_path = root_path
        self.total_experiments = total_experiments

classfication_experiment = ExperimentType("/home/xjia2/p2mss/classification_tasks", 1)
generation_experiment = ExperimentType("/home/xjia2/p2mss/generation_tasks", 3)
generation_best_experiment = ExperimentType("/home/xjia2/p2mss/generation_tasks_best", 1)

def retrieve_data(csv_path, parameter):
    temperature, input_constraint, output_constraint, epoch_number = parameter
    df = pd.read_csv(csv_path)
    query_string = f"generation_temperature == {temperature} and intput_length_constraint == {input_constraint} and output_length_constraint == {output_constraint}"
    selected_data = df.query(query_string)
    epoch_data = selected_data[f'epoch_{epoch_number}'].values
    return epoch_data

def get_validation_score(experiment_type, task, parameter):
    root_path = experiment_type.root_path
    total_experiments = experiment_type.total_experiments
    experiments_dic = {
        40: [1,2,3],
        20: [4,5,6]
    }
    subdirectories = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    count = 0
    best = 0

    generation_batch_size = parameter[-1]
    for subdirectory in subdirectories:
        if subdirectory.split("_")[1] == task and int(subdirectory.split("_")[-1]) in experiments_dic[generation_batch_size]:
            count+=1
            subdirectory_path = os.path.join(root_path, subdirectory)
            fine_tuned_score = retrieve_data(subdirectory_path+"/experiment_results.csv",parameter[:-1])
            best = max(fine_tuned_score[0]/baseline_scores[task], best)
            # best += fine_tuned_score[0]/baseline_scores[task]
            if count == total_experiments:
                return best

best_score = 0
best_parameter = None
# for classification
parameters = [(0.6, 'False', 'False', 1), (0.6, 'False', 'False', 2), (0.6, 'False', 'False', 3), (0.6, 'True', 'False', 1), (0.6, 'True', 'False', 2), (0.6, 'True', 'False', 3), (0.7, 'False', 'False', 1), (0.7, 'False', 'False', 2), (0.7, 'False', 'False', 3), (0.7, 'True', 'False', 1), (0.7, 'True', 'False', 2), (0.7, 'True', 'False', 3), (0.8, 'False', 'False', 1), (0.8, 'False', 'False', 2), (0.8, 'False', 'False', 3), (0.8, 'True', 'False', 1), (0.8, 'True', 'False', 2), (0.8, 'True', 'False', 3), (0.9, 'False', 'False', 1), (0.9, 'False', 'False', 2), (0.9, 'False', 'False', 3), (0.9, 'True', 'False', 1), (0.9, 'True', 'False', 2), (0.9, 'True', 'False', 3), (1.0, 'False', 'False', 1), (1.0, 'False', 'False', 2), (1.0, 'False', 'False', 3), (1.0, 'True', 'False', 1), (1.0, 'True', 'False', 2), (1.0, 'True', 'False', 3)]
# for generation
parameters = [(0.6, 'False', 'False', 1, 20), (0.6, 'False', 'False', 1, 40), (0.6, 'False', 'False', 2, 20), (0.6, 'False', 'False', 2, 40), (0.6, 'False', 'False', 3, 20), (0.6, 'False', 'False', 3, 40), (0.6, 'True', 'False', 1, 20), (0.6, 'True', 'False', 1, 40), (0.6, 'True', 'False', 2, 20), (0.6, 'True', 'False', 2, 40), (0.6, 'True', 'False', 3, 20), (0.6, 'True', 'False', 3, 40), (0.7, 'False', 'False', 1, 20), (0.7, 'False', 'False', 1, 40), (0.7, 'False', 'False', 2, 20), (0.7, 'False', 'False', 2, 40), (0.7, 'False', 'False', 3, 20), (0.7, 'False', 'False', 3, 40), (0.7, 'True', 'False', 1, 20), (0.7, 'True', 'False', 1, 40), (0.7, 'True', 'False', 2, 20), (0.7, 'True', 'False', 2, 40), (0.7, 'True', 'False', 3, 20), (0.7, 'True', 'False', 3, 40), (0.8, 'False', 'False', 1, 20), (0.8, 'False', 'False', 1, 40), (0.8, 'False', 'False', 2, 20), (0.8, 'False', 'False', 2, 40), (0.8, 'False', 'False', 3, 20), (0.8, 'False', 'False', 3, 40), (0.8, 'True', 'False', 1, 20), (0.8, 'True', 'False', 1, 40), (0.8, 'True', 'False', 2, 20), (0.8, 'True', 'False', 2, 40), (0.8, 'True', 'False', 3, 20), (0.8, 'True', 'False', 3, 40), (0.9, 'False', 'False', 1, 20), (0.9, 'False', 'False', 1, 40), (0.9, 'False', 'False', 2, 20), (0.9, 'False', 'False', 2, 40), (0.9, 'False', 'False', 3, 20), (0.9, 'False', 'False', 3, 40), (0.9, 'True', 'False', 1, 20), (0.9, 'True', 'False', 1, 40), (0.9, 'True', 'False', 2, 20), (0.9, 'True', 'False', 2, 40), (0.9, 'True', 'False', 3, 20), (0.9, 'True', 'False', 3, 40), (1.0, 'False', 'False', 1, 20), (1.0, 'False', 'False', 1, 40), (1.0, 'False', 'False', 2, 20), (1.0, 'False', 'False', 2, 40), (1.0, 'False', 'False', 3, 20), (1.0, 'False', 'False', 3, 40), (1.0, 'True', 'False', 1, 20), (1.0, 'True', 'False', 1, 40), (1.0, 'True', 'False', 2, 20), (1.0, 'True', 'False', 2, 40), (1.0, 'True', 'False', 3, 20), (1.0, 'True', 'False', 3, 40)]
classification_tasks = ["task202", "task199", "task1388", "task201", "task190", "task935", "task1386", "task1554", "task738", "task1344", "task1385", "task1529", "task200", "task1612", "task937", "task1516", "task020", "task1615"]
generation_tasks = ["task039", "task281", "task121", "task1195", "task034", "task1622", "task1562", "task671", "task1345", "task035", "task1659", "task569","task1631", "task1557", "task036"]
# 没有继续跑的 "task1540",  "task620",  "task613", "task677", "task1356", "task957",  "task1598", 

# Define a dictionary to store baseline validation scores for each task
baseline_scores = {
    "task202": 0.21366666666666667,
    "task199": 0.4176666666666667,
    "task1388": 0.4533333333333333,
    "task201": 0.244,
    "task190": 0.212,
    "task935": 0.694,
    "task1386": 0.35918367346938773,
    "task1554": 0.713,
    "task738": 0.838,
    "task1344": 0.741, 
    "task1385": 0.4311111111111111, 
    "task1529": 0.78, 
    "task200": 0.729,
    "task1612": 0.6025, 
    "task937": 0.659, 
    "task1516": 0.6057142857142858, 
    "task020": 0.86, 
    "task1615": 0.638,
    'task039': 0.41750204802155055, 
    'task281': 0.4923867647664332, 
    'task121': 0.5836333975037971, 
    'task1195': 0.8434063049996244, 
    'task034': 0.9542277289816776, 
    'task1622': 0.8735489511140455, 
    'task1562': 0.6641334172434672, 
    'task671': 0.655367618110219, 
    'task1345': 0.5282595788877378, 
    'task035': 0.9170775810808315, 
    'task1659': 0.45471142444914453, 
    'task1540': 0.49899970285363515, 
    'task1356': 0.4605926122521061, 
    'task569': 0.46845037734437633, 
    'task957': 0.5879953692728069, 
    'task1598': 0.5304212892800535, 
    'task1631': 0.9777259563702011, 
    'task677': 0.43151514017389464, 
    'task1557': 0.8704846656227533, 
    'task036': 0.4897276973745361, 
    'task613': 0.4630338019228093, 
    'task620': 0.23528267296145397
}


score_recording = {}
for parameter in parameters:
    if parameter[-1]==40:
        continue
    parameter_score_sum = 0
    for task in generation_tasks:
        best_score_task = get_validation_score(generation_best_experiment, task, parameter)
        parameter_score_sum += best_score_task
    if parameter_score_sum >= best_score:
        best_score = parameter_score_sum
        best_parameter = parameter
    score_recording[parameter] = parameter_score_sum
print(best_parameter) 
print(score_recording)