# tasks

# task_id, validation_acc, test_acc, baseline_validation_acc, baseline_test_acc

tasks = [
    ("039", 0.3984590845557454, 0.39236909332114944, 0.3196946463457281, 0.3186090096156368),
    ("281", 0.4906044314806828, 0.4799966461370743, 0.4045382338758414, 0.3935494293992201),
    ("121", 0.6335851215771953, 0.5631263520833625, 0.5944038995840331, 0.5974003818903908),
    ("1195", 0.8602194928156706, 0.8598463598278795, 0.8754831322999768, 0.8839160573830065),
    ("034", 0.960356841733781, 0.960356841733781, 0.8906854245139225, 0.8935329308204496),
    ("1622", 0.8232145341816213, 0.8360647128185769, 0.8746147420704412, 0.872073399387891),
    ("1562", 0.6980046599035434, 0.6272727431227658, 0.6405707744714424, 0.6206518431595964),
    ("671", 0.6913801932903522, 0.68824838493264, 0.5811770053120827, 0.579990306709856),
    ("1345", 0.538699242646657, 0.5473119083648297, 0.5071687971247202, 0.5179178664637611),
    ("035", 0.9375729180322491, 0.9398058828821471, 0.90859511922213, 0.9124373032332922),
    ("1659", 0.5749928997338861, 0.5794975605255527, 0.5573634380731896, 0.5651243389546297),
    ("1540", 0.46467592196583, 0.4676202666149862, 0.4756219748033306, 0.48153662690099985),
    ("1356", 0.4157955982599913, 0.41892777618287325, 0.4200315679576199, 0.43198310438345927),
     ("569", 0.42739753371110406, 0.43098605033093745, 0.46176260370051314, 0.48053118437046777),
     ("957", 0.5551255269300266, 0.5515100195712258, 0.587197788966534, 0.5836113437431252),
     ("1598", 0.5028006534386302, 0.5004954939334921, 0.5175260987976953, 0.5182558164838015),
     ("1631", 0.9441666676476308, 0.9442888495105336, 0.9187278189839666, 0.9150836016826712),
     ("677", 0.41946911482446636, 0.40826220251885603, 0.4246117301157084, 0.4167249710186653),
     ("1557", 0.9092477241403365, 0.9146926051087805, 0.8747644649506419, 0.861743368665078),
     ("036", 0.5081344022650549, 0.5135661473349648, 0.4337328248936478, 0.4385026360720763),
     ("613", 0.4126600518818201, 0.4148191917407171, 0.3801760803186837, 0.3997454577450567),
]

# Classify tasks and calculate relative improvements/decreases
improved_tasks = []
decreased_tasks = []
overfit_tasks = []
same_tasks = []

# Initialize variables to store the sum of relative improvements/decreases
sum_relative_improvement_validation = 0
sum_relative_improvement_test = 0
sum_relative_decrease_validation = 0
sum_relative_decrease_test = 0

for each in tasks:
    task_id, validation_acc, test_acc, baseline_validation_acc, baseline_test_acc = each
    abs_improvement_val = validation_acc - baseline_validation_acc
    abs_improvement_test = test_acc - baseline_test_acc

    # Calculate relative improvements/decreases
    rel_improvement_val = abs_improvement_val / baseline_validation_acc
    rel_improvement_test = abs_improvement_test / baseline_test_acc

    if abs_improvement_val > 0.01 and abs_improvement_test > 0.01:
        improved_tasks.append(each)
        sum_relative_improvement_validation += rel_improvement_val
        sum_relative_improvement_test += rel_improvement_test
    elif abs_improvement_val < -0.01 and abs_improvement_test < -0.01:
        decreased_tasks.append(each)
        sum_relative_decrease_validation += -rel_improvement_val
        sum_relative_decrease_test += -rel_improvement_test
    elif abs_improvement_val > 0.01 and abs_improvement_test < -0.01:
        overfit_tasks.append(each)
    else:
        same_tasks.append(each)

print(len(improved_tasks))
print(len(decreased_tasks))
print(len(overfit_tasks))
print(len(same_tasks))

# Calculate and print the average relative improvements for improved tasks
if improved_tasks:
    avg_rel_improvement_val = (sum_relative_improvement_validation / len(improved_tasks)) * 100
    avg_rel_improvement_test = (sum_relative_improvement_test / len(improved_tasks)) * 100
    print(f"Average Relative Improvement on Validation Set: {avg_rel_improvement_val:.2f}%")
    print(f"Average Relative Improvement on Test Set: {avg_rel_improvement_test:.2f}%")

# Calculate and print the average relative decreases for decreased tasks
if decreased_tasks:
    avg_rel_decrease_val = (sum_relative_decrease_validation / len(decreased_tasks)) * 100
    avg_rel_decrease_test = (sum_relative_decrease_test / len(decreased_tasks)) * 100
    print(f"Average Relative Decrease on Validation Set: {avg_rel_decrease_val:.2f}%")
    print(f"Average Relative Decrease on Test Set: {avg_rel_decrease_test:.2f}%")


# Classify tasks and calculate relative improvements/decreases for each task
improved_tasks = []
decreased_tasks = []
overfit_tasks = []
same_tasks = []

# Initialize variables to store the sum of relative improvements/decreases across all tasks
sum_relative_improvement_validation_all = 0
sum_relative_improvement_test_all = 0

for each in tasks:
    task_id, validation_acc, test_acc, baseline_validation_acc, baseline_test_acc = each
    abs_improvement_val = validation_acc - baseline_validation_acc
    abs_improvement_test = test_acc - baseline_test_acc

    # Calculate relative improvements/decreases for each task
    rel_improvement_val = abs_improvement_val / baseline_validation_acc
    rel_improvement_test = abs_improvement_test / baseline_test_acc

    # Sum the relative improvements/decreases across all tasks
    sum_relative_improvement_validation_all += rel_improvement_val
    sum_relative_improvement_test_all += rel_improvement_test

    # Classify tasks based on absolute differences
    if abs_improvement_val > 0.01 and abs_improvement_test > 0.01:
        improved_tasks.append(each)
    elif abs_improvement_val < -0.01 and abs_improvement_test < -0.01:
        decreased_tasks.append(each)
    elif abs_improvement_val > 0.01 and abs_improvement_test < -0.01:
        overfit_tasks.append(each)
    else:
        same_tasks.append(each)

# Calculate and print the overall average relative improvements across all tasks
avg_rel_improvement_validation_all = (sum_relative_improvement_validation_all / len(tasks)) * 100
avg_rel_improvement_test_all = (sum_relative_improvement_test_all / len(tasks)) * 100
print(f"Average Relative Improvement on Validation Set for All Tasks: {avg_rel_improvement_validation_all:.2f}%")
print(f"Average Relative Improvement on Test Set for All Tasks: {avg_rel_improvement_test_all:.2f}%")
