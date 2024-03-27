import pandas as pd
import random

def retrieve_data(csv_path, parameter):
    temperature, input_constraint, output_constraint, epoch_number = parameter
    df = pd.read_csv(csv_path)
    query_string = f"generation_temperature == {temperature} and intput_length_constraint == {input_constraint} and output_length_constraint == {output_constraint}"
    selected_data = df.query(query_string)
    epoch_data = selected_data[f'epoch_{epoch_number}'].values
    return epoch_data[0]

def get_validation_score(generation, task, parameter):
    if generation:
        root_path = f'/home/azureuser/p2mss/p2mss/generation_11/NI_{task}_exp_11'
    else:
        root_path = f'/home/azureuser/p2mss/p2mss/classification_14/NI_{task}_exp_14'
    fine_tuned_score = retrieve_data(root_path+"/experiment_results.csv", parameter)
    return fine_tuned_score
    # best += fine_tuned_score[0]/baseline_scores[task]


best_score = 0
# for classification
classification_parameters = [
    (0.6, 'True', 'False', 1), 
    (0.6, 'True', 'False', 2), 
    (0.6, 'True', 'False', 3), 
    (0.7, 'True', 'False', 1), 
    (0.7, 'True', 'False', 2), 
    (0.7, 'True', 'False', 3), 
    (0.8, 'True', 'False', 1), 
    (0.8, 'True', 'False', 2), 
    (0.8, 'True', 'False', 3), 
    (0.9, 'True', 'False', 1), 
    (0.9, 'True', 'False', 2), 
    (0.9, 'True', 'False', 3), 
    (1.0, 'True', 'False', 1), 
    (1.0, 'True', 'False', 2), 
    (1.0, 'True', 'False', 3)
]

# for generation
generation_parameters = [
    (0.6, 'True', 'True', 1), (0.6, 'True', 'True', 2), (0.6, 'True', 'True', 3), 
    (0.7, 'True', 'True', 1), (0.7, 'True', 'True', 2), (0.7, 'True', 'True', 3), 
    (0.8, 'True', 'True', 1), (0.8, 'True', 'True', 2), (0.8, 'True', 'True', 3), 
    (0.9, 'True', 'True', 1), (0.9, 'True', 'True', 2), (0.9, 'True', 'True', 3), 
    (1.0, 'True', 'True', 1), (1.0, 'True', 'True', 2), (1.0, 'True', 'True', 3)
]

classification_tasks = ["task190", "task199", "task200", "task738", "task937", "task1385", "task1386"]
generation_tasks = ["task039", "task036", "task1195", "task121"] 

classification_test_tasks = ["task1615", "task284", "task329", "task346", "task1516", "task1529", "task1612"]
generation_test_tasks = ["task281",  "task1622", "task1345", "task1562"]



# Define a dictionary to store baseline validation scores for each task
old_baseline_scores = {
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

baseline_scores = {
    "task190": 0.27003930951974, 
    "task199": 0.316239316239316, 
    "task200": 0.46752923612839, 
    "task738": 0.596424231904183, 
    "task937": 0.471111111111111, 
    "task1385": 0.331838565022421, 
    "task1386": 0.339664804469273, 
    "task1516": 0.1765625, 
    "task1529": 0.0856201975850713, 
    "task1612": 0.51358024691358,
    "task1615": 0.00555898702903026,
    "task284": 0.900649794801641,
    "task329": 0.291063404892661,
    "task346": 0.351744186046511,

    "task121": 0.492259181898013, 
    "task039": 0.122302578999944, 
    "task036": 0.188900032990357, 
    "task1195": 0.436834178312443, 
    "task1345": 0.371234390180198,
    "task1562": 0.310781117123961,
    "task281": 0.411776622489333,
    "task1622": 0.444840217463775
}

print('generation tasks')
score_recording = {}
# all_generation = generation_tasks + generation_test_tasks
# random_items = random.sample(all_generation, 4)
for parameter in generation_parameters:
    parameter_score = []
    for task in generation_tasks:
        task_score = get_validation_score(generation=True, task=task, parameter=parameter)
        task_score -= baseline_scores[task]
        parameter_score.append(task_score)
    score_recording[parameter] = min(parameter_score)
param_with_max_improve = max(score_recording, key=lambda k: score_recording[k])
# test_score_recording = {}
# 
# for task in all_generation:
#     task_score = get_validation_score(generation=True, task=task, parameter=param_with_max_improve)
#     test_score_recording[task] = task_score
print(param_with_max_improve) 
# print(score_recording)
# print(test_score_recording)

print('classification tasks')
score_recording = {}
# all_classification = classification_tasks + classification_test_tasks
# random_items = random.sample(all_classification, 4)
for parameter in classification_parameters:
    parameter_score = []
    for task in classification_tasks:
        task_score = get_validation_score(generation=False, task=task, parameter=parameter)
        task_score -= baseline_scores[task]
        parameter_score.append(task_score)
    score_recording[parameter] = min(parameter_score)
param_with_max_improve = max(score_recording, key=lambda k: score_recording[k])
# test_score_recording = {}
# 
# for task in all_classification:
#     task_score = get_validation_score(generation=False, task=task, parameter=param_with_max_improve)
#     test_score_recording[task] = task_score
print(param_with_max_improve) 
# print(score_recording)
# print(test_score_recording)