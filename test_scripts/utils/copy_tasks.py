import json
from tasks import Task

# 读取JSON文件
with open('/home/cyzhao/main/NI_tasks/tasks.json', 'r', encoding='utf-8') as json_file:
    datas = json.load(json_file)
    for data in datas:
        print(data['task_name'])
        if data['task_name'] == 'task1615':
            # 将JSON数据实例化为Task类
            task_instance = Task(
                task_instruction=data["task_instruction"],
                task_name=data["task_name"],
                examples=data["examples"],
                expected_content=data["expected_content"],
                optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore", "Hence"],
                metric=data["metric"],
                labels=data["labels"],
                is_classification=True,
                extraction_examples=[]
            )

            # 将Task实例写入Python文件
            with open('/home/cyzhao/main/test_scripts/utils/tasks.py', 'a', encoding='utf-8') as python_file:
                python_file.write(f'''
{task_instance.task_name} = Task(
    task_instruction="""{task_instance.task_instruction}""".strip(),
    task_name="{task_instance.task_name}",
    examples="""{task_instance.examples}""".strip(),
    expected_content="""{task_instance.expected_content}
    """.strip(),
    optional_list={task_instance.optional_list},
    metric="{task_instance.metric}",
    labels={task_instance.labels},
    is_classification={task_instance.is_classification},
    extraction_examples={task_instance.extraction_examples}
)
            ''')
