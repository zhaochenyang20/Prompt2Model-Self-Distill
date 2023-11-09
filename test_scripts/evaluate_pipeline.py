import os
from pathlib import Path

cuda_condition = "CUDA_VISIBLE_DEVICES=0"

ckpt_path = Path("/home/cyzhao/ckpt")

for each in [each for each in os.listdir(ckpt_path) if "_" in each]:
    os.system(
        f"{cuda_condition} python ./test_scripts/test_squad.py --model_name {each}"
    )
