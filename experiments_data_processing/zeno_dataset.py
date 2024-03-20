from zeno_client import ZenoClient, ZenoMetric
import pandas as pd
import os
from datasets import load_from_disk
import json

client = ZenoClient("zen_fcbJnbmEJglYTu8PSRnWCIWp99yKtt1V5YsAYRo_0Ls")

# tasks = ['121', '039', '036', '281', '1195', '1345', '1562', '1622']

tasks = ['190', '199', '200', '284', '329', '346', '738', '937', '1385', '1386', '1516', '1529', '1612', '1615']


for task in tasks:

    project = client.create_project(
        name=f"Classification Data {task}",
        view="text-classification",
        metrics=[]
    )

    root_path = f'/home/azureuser/p2mss/p2mss/classification_14/NI_task{task}_exp_14'
    best_validation_result_path = os.path.join(root_path, 'best_validation_result.json')
    with open(best_validation_result_path, 'r') as file:
        results = json.load(file)
    evaluate_result_path = '/'.join(results['evaluate_result_path'].split('/')[:-1])
    dataset_path = os.path.join(evaluate_result_path, 'dataset') 
    dataset = load_from_disk(dataset_path)
    df = dataset.to_pandas()
    df['id'] = range(len(df))
    df["input_length"] = df["input_col"].str.len()
    df["ground_truth_output_length"] = df["output_col"].str.len()
    project.upload_dataset(df, id_column="id", data_column="input_col", label_column="output_col")