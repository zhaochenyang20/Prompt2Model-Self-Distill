import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
csv_file_path = 'dataset_summary.csv'
df = pd.read_csv(csv_file_path)

# 将字符串列转换为数值型，以便进行分析

df['generation_temperature'] = pd.to_numeric(df['generation_temperature'], errors='coerce')
df['input_length_constraint'] = pd.to_numeric(df['input_length_constraint'], errors='coerce')
df['output_length_constraint'] = pd.to_numeric(df['output_length_constraint'], errors='coerce')
df['num_rows'] = pd.to_numeric(df['num_rows'], errors='coerce')

# 显示基本统计量
print(df.describe())

# 计算并显示相关系数矩阵
print(df.corr())

# 数据可视化
# 散点图矩阵
sns.pairplot(df)
plt.show()

# 箱形图可视化generation_temperature与num_rows的关系
sns.boxplot(x='generation_temperature', y='num_rows', data=df)
plt.show()

# 箱形图可视化input_length_constraint与num_rows的关系
sns.boxplot(x='input_length_constraint', y='num_rows', data=df)
plt.show()

# 箱形图可视化output_length_constraint与num_rows的关系
sns.boxplot(x='output_length_constraint', y='num_rows', data=df)
plt.show()
