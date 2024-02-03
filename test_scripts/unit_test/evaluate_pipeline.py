import os
from pathlib import Path
from utils.path import ROOT

cuda_condition = "CUDA_VISIBLE_DEVICES=0"

ckpt_path = Path(ROOT+"/ckpt")

for each in [each for each in os.listdir(ckpt_path) if "_" in each]:
    os.system(f"{cuda_condition} python ./unit_test/test_squad.py --model_name {each}")
