def compare_files(file1_path, file2_path):
    # 逐行比较两个文件
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines_file1 = file1.readlines()
        lines_file2 = file2.readlines()

        # 获取文件行数
        len_file1 = len(lines_file1)
        len_file2 = len(lines_file2)

        # 比较两个文件的行数
        min_len = min(len_file1, len_file2)

        for i in range(min_len):
            # 逐行比较内容
            if lines_file1[i] != lines_file2[i]:
                print(f"Difference in line {i + 1}:\nFile 1: {lines_file1[i]}File 2: {lines_file2[i]}")

        # 打印剩余行（如果有）
        if len_file1 > len_file2:
            for i in range(min_len, len_file1):
                print(f"Extra line in File 1, line {i + 1}: {lines_file1[i]}")
        elif len_file2 > len_file1:
            for i in range(min_len, len_file2):
                print(f"Extra line in File 2, line {i + 1}: {lines_file2[i]}")
        print('they are same')

# 用法示例
task_number = '1615'
temperatures = [0.6, 0.7, 0.8, 0.9, 1.0]
pairs = [
    ('False', 'False'),
    ('True', 'False')
]
root_path = "/home/cyzhao"
res = []
experiment_number_1 = 4
experiment_number_2 = 1
for generation_temperature in temperatures:
    for (input_length_constraint,output_length_constraint) in pairs:  
        for epoch in range(1,4):
            res.append((generation_temperature, input_length_constraint, output_length_constraint, epoch))
print(res) 
        # file1_path = f'{root_path}/NI_task{task_number}_exp_{experiment_number_1}/task{task_number}_{generation_temperature}_{input_length_constraint}_{output_length_constraint}_{experiment_number_1}/dataset.txt'
        # file2_path = f'{root_path}/NI_task{task_number}_exp_{experiment_number_2}/task{task_number}_{generation_temperature}_{input_length_constraint}_{output_length_constraint}_{experiment_number_2}/dataset.txt'
        # compare_files(file1_path, file2_path)
