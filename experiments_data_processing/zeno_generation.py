from zeno_client import ZenoClient, ZenoMetric
import pandas as pd
from datasets import load_from_disk

client = ZenoClient("zen_fcbJnbmEJglYTu8PSRnWCIWp99yKtt1V5YsAYRo_0Ls")

tasks = ['121', '039', '036', '281', '1195', '1345', '1562', '1622']
for task in tasks:

    project = client.create_project(
        name=f"Self-guild Generation Task Data Analysis Task {task}",
        view="text-classification",
        metrics=[]
    )

    test_dataset = load_from_disk(f"/home/azureuser/p2mss/prompt2model_test/testdataset/NI/test/task{task}")
    df_test = test_dataset.to_pandas()
    df_test['id'] = range(len(df_test))
    df_test["input_length"] = df_test["input_col"].str.len()
    df_test["ground_truth_output_length"] = df_test["output_col"].str.len()
    project.upload_dataset(df_test, id_column="id", data_column="input_col", label_column="output_col")
    
    def lcs_length_dp(x, y):
        """Compute the length of the longest common subsequence between two strings using dynamic programming."""
        m, n = len(x), len(y)
        dp_table = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    dp_table[i][j] = 0
                elif x[i - 1] == y[j - 1]:
                    dp_table[i][j] = dp_table[i - 1][j - 1] + 1
                else:
                    dp_table[i][j] = max(dp_table[i - 1][j], dp_table[i][j - 1])

        return dp_table[m][n]


    def rouge_l_score(row):
        model_output = row['model_output']
        groud_truth = row['groud_truth']
        lcs = lcs_length_dp(groud_truth, model_output)
        if lcs == 0:
            return 0
        precision = lcs / len(model_output)
        recall = lcs / len(groud_truth)
        f_measure = (2 * precision * recall) / (precision + recall)
        return f_measure



    baseline_results = load_from_disk(f'/home/azureuser/p2mss/p2mss/baseline_generated_data/20240310_test_task{task}')
    df_baseline = baseline_results.to_pandas()
    df_baseline['id'] = range(len(df_baseline))
    df_baseline['correct'] = df_baseline.apply(rouge_l_score, axis=1)
    df_baseline["baseline_output_length"] = df_baseline["model_output"].str.len()
    project.upload_system(df_baseline, name="baseline", id_column="id", output_column="model_output")
    
    finetuned_results = load_from_disk(f'/home/azureuser/p2mss/p2mss/generation_11/NI_task{task}_exp_11/best_ckpt_generated_content')
    df_finetune = finetuned_results.to_pandas()
    df_finetune['id'] = range(len(df_finetune))
    df_finetune['correct'] = df_baseline.apply(rouge_l_score, axis=1)
    df_finetune["finetune_output_length"] = df_finetune["model_output"].str.len()
    project.upload_system(df_finetune, name="finetune", id_column="id", output_column="model_output")  

