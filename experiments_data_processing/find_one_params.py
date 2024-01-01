import os
import pandas as pd

def retrieve_data(csv_path, parameter):
    temperature, input_constraint, output_constraint, epoch_number = parameter
    df = pd.read_csv(csv_path)
    query_string = f"generation_temperature == {temperature} and intput_length_constraint == {input_constraint} and output_length_constraint == {output_constraint}"
    selected_data = df.query(query_string)
    epoch_data = selected_data[f'epoch_{epoch_number}'].values
    return epoch_data

def get_validation_score(task, parameter):
    root_path = "/home/cyzhao/classification_tasks"
    subdirectories = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    for subdirectory in subdirectories:
        if subdirectory.split("_")[1] == task:
            subdirectory_path = os.path.join(root_path, subdirectory)
            fine_tuned_score = retrieve_data(subdirectory_path+"/experiment_results.csv",parameter)
            return fine_tuned_score/baseline_scores[task]

best_score = 0
best_parameter = None
parameters = [(0.6, 'False', 'False', 1), (0.6, 'False', 'False', 2), (0.6, 'False', 'False', 3), (0.6, 'True', 'False', 1), (0.6, 'True', 'False', 2), (0.6, 'True', 'False', 3), (0.7, 'False', 'False', 1), (0.7, 'False', 'False', 2), (0.7, 'False', 'False', 3), (0.7, 'True', 'False', 1), (0.7, 'True', 'False', 2), (0.7, 'True', 'False', 3), (0.8, 'False', 'False', 1), (0.8, 'False', 'False', 2), (0.8, 'False', 'False', 3), (0.8, 'True', 'False', 1), (0.8, 'True', 'False', 2), (0.8, 'True', 'False', 3), (0.9, 'False', 'False', 1), (0.9, 'False', 'False', 2), (0.9, 'False', 'False', 3), (0.9, 'True', 'False', 1), (0.9, 'True', 'False', 2), (0.9, 'True', 'False', 3), (1.0, 'False', 'False', 1), (1.0, 'False', 'False', 2), (1.0, 'False', 'False', 3), (1.0, 'True', 'False', 1), (1.0, 'True', 'False', 2), (1.0, 'True', 'False', 3)]
tasks = ["task202", "task199", "task1388", "task201", "task190", "task935", "task1386", "task1554", "task738", "task1344", "task1385", "task1529", "task200", "task1612", "task937", "task1516", "task020", "task1615"]

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
    "task1615": 0.638
}


score_recording = {}
for parameter in parameters:
    parameter_score_sum = 0
    for task in tasks:
        best_score_task = get_validation_score(task, parameter)
        parameter_score_sum += best_score_task
    if parameter_score_sum >= best_score:
        best_score = parameter_score_sum
        best_parameter = parameter
    score_recording[parameter] = parameter_score_sum
print(best_parameter) 
print(score_recording)