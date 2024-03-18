import shutil
import os

def copy_dir_contents(src_dir, dst_dir):
    """
    复制src_dir目录下的所有内容到dst_dir目录。
    
    参数:
    src_dir (str): 源目录路径。
    dst_dir (str): 目标目录路径。
    """
    # 确保目标目录存在
        
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        if os.path.isdir(src_path):
            # 如果是文件夹，则递归复制
            shutil.copytree(src_path, dst_path)
        else:
            # 如果是文件，则直接复制
            shutil.copy2(src_path, dst_path)

# 指定源目录和目标目录的路径
src_directory = '/home/azureuser/p2mss/p2mss/NI_task281_exp_11'
dst_directory = '/home/azureuser/p2mss/p2mss/generation_11/NI_task281_exp_11'

# 调用函数执行复制操作
copy_dir_contents(src_directory, dst_directory)
