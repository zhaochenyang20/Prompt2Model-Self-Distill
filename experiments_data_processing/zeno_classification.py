from zeno_client import ZenoClient, ZenoMetric
import pandas as pd
from datasets import load_from_disk

client = ZenoClient("zen_fcbJnbmEJglYTu8PSRnWCIWp99yKtt1V5YsAYRo_0Ls")

tasks = ['190', '199', '200', '284', '329', '346', '738', '937', '1385', '1386', '1516', '1529', '1612', '1615']

for task in tasks:

    project = client.create_project(
        name=f"Classification Task {task}",
        view="text-classification",
        metrics=[
            ZenoMetric(name="accuracy", type="mean", columns=["correct"])
        ]
    )

    test_dataset = load_from_disk(f"/home/azureuser/p2mss/prompt2model_test/testdataset/NI/test/task{task}")
    df_test = test_dataset.to_pandas()
    df_test['id'] = range(len(df_test))
    df_test["input_length"] = df_test["input_col"].str.len()
    project.upload_dataset(df_test, id_column="id", data_column="input_col", label_column="output_col")

    def check_mutual_inclusion(row):
        model_output = row['model_output']
        groud_truth = row['groud_truth']
        if model_output == '':
            return 0
        return int(model_output in groud_truth or groud_truth in model_output)


    baseline_results = load_from_disk(f'/home/azureuser/p2mss/p2mss/baseline_generated_data/20240310_test_task{task}')
    df_baseline = baseline_results.to_pandas()
    df_baseline['id'] = range(len(df_baseline))
    df_baseline['correct'] = df_baseline.apply(check_mutual_inclusion, axis=1)
    project.upload_system(df_baseline, name="baseline", id_column="id", output_column="model_output")
    
    finetuned_results = load_from_disk(f'/home/azureuser/p2mss/p2mss/classification_14/NI_task{task}_exp_14/best_ckpt_generated_content')
    df_finetune = finetuned_results.to_pandas()
    df_finetune['id'] = range(len(df_finetune))
    df_finetune['correct'] = df_finetune.apply(check_mutual_inclusion, axis=1)
    project.upload_system(df_finetune, name="finetune", id_column="id", output_column="model_output")  

