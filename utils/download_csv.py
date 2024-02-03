from datasets import load_from_disk
from utils.path import ROOT

dataset = load_from_disk(ROOT+"/base_vicuna_squad")

dataset = dataset.rename_column("model_input", "input")
dataset = dataset.rename_column("model_output", "raw_vicuna_output")
dataset = dataset.rename_column("groud_truth", "ground_truth")

dataset.to_csv("SQuAD_raw_vicuna.csv")
