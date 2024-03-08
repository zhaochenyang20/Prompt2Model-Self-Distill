from datasets import load_from_disk
from collections import Counter
import os
# go through all 8 rank experiments
tasks = ['task199', 'task200', 'task738', 'task937', 'task1385', 'task1386', 'task1516', 'task1529', 'task1612', 'task1615', 'task190']

for task in tasks:
    # generated dataset
    print(task)

    path = f'/home/azureuser/p2mss/p2mss/NI_{task}_exp_8'
    for folder_name in os.listdir(path):
        if folder_name.startswith(task):
            dataset_path = os.path.join(path, folder_name, 'dataset')
            dataset = load_from_disk(dataset_path)
            outputs = Counter(dataset['output_col'])
            for output, num in outputs.items():
                print(f'generated dataset {output}: {num}')

    # evaluation dataset
    path = f'/home/azureuser/p2mss/prompt2model_test/testdataset/NI/eval/{task}'
    dataset = load_from_disk(path)
    outputs = Counter(dataset['output_col'])
    for output, num in outputs.items():
        print(f'validation dataset for {output}: {num}')    
    
    # test dataset
    path = f'/home/azureuser/p2mss/prompt2model_test/testdataset/NI/test/{task}'
    dataset = load_from_disk(path)
    outputs = Counter(dataset['output_col'])
    for output, num in outputs.items():
        print(f'test dataset for {output}: {num}')          





        

# task199
# generated dataset no: 63
# generated dataset yes: 63
# generated dataset yes: 62
# generated dataset no: 62
# generated dataset yes: 80
# generated dataset no: 80
# generated dataset no: 66
# generated dataset yes: 66
# generated dataset no: 65
# generated dataset yes: 65
# generated dataset yes: 74
# generated dataset no: 74
# generated dataset no: 96
# generated dataset yes: 96
# generated dataset no: 90
# generated dataset yes: 90
# generated dataset yes: 84
# generated dataset no: 84
# generated dataset no: 65
# generated dataset yes: 65
# validation dataset for no: 962
# validation dataset for yes: 2038
# test dataset for yes: 1964
# test dataset for no: 1036
# task200
# generated dataset 2: 68
# generated dataset 1: 68
# generated dataset 3: 68
# generated dataset 2: 78
# generated dataset 1: 78
# generated dataset 3: 78
# generated dataset 1: 72
# generated dataset 2: 72
# generated dataset 3: 72
# generated dataset 2: 57
# generated dataset 3: 57
# generated dataset 1: 57
# generated dataset 2: 43
# generated dataset 1: 43
# generated dataset 3: 43
# generated dataset 3: 86
# generated dataset 1: 86
# generated dataset 2: 86
# generated dataset 3: 97
# generated dataset 2: 97
# generated dataset 1: 97
# generated dataset 2: 77
# generated dataset 1: 77
# generated dataset 3: 77
# generated dataset 1: 73
# generated dataset 3: 73
# generated dataset 2: 73
# generated dataset 2: 61
# generated dataset 1: 61
# generated dataset 3: 61
# validation dataset for 3: 337
# validation dataset for 1: 348
# validation dataset for 2: 315
# test dataset for 1: 340
# test dataset for 2: 352
# test dataset for 3: 308
# task738
# generated dataset support: 136
# generated dataset undermine: 136
# generated dataset support: 127
# generated dataset undermine: 127
# generated dataset undermine: 112
# generated dataset support: 112
# generated dataset undermine: 40
# generated dataset support: 40
# generated dataset support: 119
# generated dataset undermine: 119
# generated dataset undermine: 46
# generated dataset support: 46
# generated dataset support: 140
# generated dataset undermine: 140
# generated dataset undermine: 26
# generated dataset support: 26
# generated dataset undermine: 46
# generated dataset support: 46
# generated dataset undermine: 35
# generated dataset support: 35
# validation dataset for support: 548
# validation dataset for undermine: 452
# test dataset for support: 560
# test dataset for undermine: 440
# task937
# generated dataset weakener: 16
# generated dataset strengthener: 16
# generated dataset weakener: 13
# generated dataset strengthener: 13
# generated dataset weakener: 22
# generated dataset strengthener: 22
# generated dataset weakener: 26
# generated dataset strengthener: 26
# generated dataset weakener: 81
# generated dataset strengthener: 81
# generated dataset strengthener: 31
# generated dataset weakener: 31
# generated dataset weakener: 66
# generated dataset strengthener: 66
# generated dataset weakener: 5
# generated dataset strengthener: 5
# generated dataset weakener: 52
# generated dataset strengthener: 52
# generated dataset strengthener: 43
# generated dataset weakener: 43
# validation dataset for strengthener: 1579
# validation dataset for weakener: 1421
# test dataset for strengthener: 1552
# test dataset for weakener: 1448
# task1385
# generated dataset Neutral: 59
# generated dataset Entailment: 59
# generated dataset Contradiction: 59
# generated dataset Contradiction: 34
# generated dataset Neutral: 34
# generated dataset Entailment: 34
# generated dataset Contradiction: 49
# generated dataset Entailment: 49
# generated dataset Neutral: 49
# generated dataset Neutral: 16
# generated dataset Contradiction: 16
# generated dataset Entailment: 16
# generated dataset Contradiction: 29
# generated dataset Neutral: 29
# generated dataset Entailment: 29
# generated dataset Contradiction: 32
# generated dataset Entailment: 32
# generated dataset Neutral: 32
# generated dataset Neutral: 17
# generated dataset Contradiction: 17
# generated dataset Entailment: 17
# generated dataset Contradiction: 58
# generated dataset Neutral: 58
# generated dataset Entailment: 58
# generated dataset Entailment: 45
# generated dataset Contradiction: 45
# generated dataset Neutral: 45
# generated dataset Contradiction: 22
# generated dataset Neutral: 22
# generated dataset Entailment: 22
# validation dataset for Contradiction: 152
# validation dataset for Entailment: 143
# validation dataset for Neutral: 155
# test dataset for Neutral: 143
# test dataset for Contradiction: 152
# test dataset for Entailment: 155
# task1386
# generated dataset Neutral: 46
# generated dataset Contradiction: 46
# generated dataset Entailment: 46
# generated dataset Contradiction: 42
# generated dataset Neutral: 42
# generated dataset Entailment: 42
# generated dataset Contradiction: 34
# generated dataset Neutral: 34
# generated dataset Entailment: 34
# generated dataset Neutral: 59
# generated dataset Entailment: 59
# generated dataset Contradiction: 59
# generated dataset Neutral: 39
# generated dataset Contradiction: 39
# generated dataset Entailment: 39
# generated dataset Contradiction: 55
# generated dataset Neutral: 55
# generated dataset Entailment: 55
# generated dataset Neutral: 25
# generated dataset Entailment: 25
# generated dataset Contradiction: 25
# generated dataset Entailment: 36
# generated dataset Neutral: 36
# generated dataset Contradiction: 36
# generated dataset Neutral: 41
# generated dataset Contradiction: 41
# generated dataset Entailment: 41
# generated dataset Contradiction: 30
# generated dataset Neutral: 30
# generated dataset Entailment: 30
# validation dataset for Contradiction: 175
# validation dataset for Neutral: 159
# validation dataset for Entailment: 156
# test dataset for Neutral: 167
# test dataset for Entailment: 173
# test dataset for Contradiction: 150
# task1516
# generated dataset neutral: 44
# generated dataset positive: 44
# generated dataset negated: 44
# generated dataset negated: 32
# generated dataset positive: 32
# generated dataset neutral: 32
# generated dataset neutral: 31
# generated dataset positive: 31
# generated dataset negated: 31
# generated dataset positive: 15
# generated dataset negated: 15
# generated dataset neutral: 15
# generated dataset neutral: 26
# generated dataset negated: 26
# generated dataset positive: 26
# generated dataset neutral: 37
# generated dataset negated: 37
# generated dataset positive: 37
# generated dataset positive: 80
# generated dataset negated: 80
# generated dataset neutral: 80
# generated dataset positive: 20
# generated dataset negated: 20
# generated dataset neutral: 20
# generated dataset negated: 41
# generated dataset positive: 41
# generated dataset neutral: 41
# generated dataset negated: 34
# generated dataset neutral: 34
# generated dataset positive: 34
# validation dataset for positive: 113
# validation dataset for negated: 119
# validation dataset for neutral: 118
# test dataset for neutral: 115
# test dataset for positive: 120
# test dataset for negated: 115
# task1529
# generated dataset entails: 104
# generated dataset neutral: 104
# generated dataset entails: 138
# generated dataset neutral: 138
# generated dataset neutral: 142
# generated dataset entails: 142
# generated dataset neutral: 121
# generated dataset entails: 121
# generated dataset neutral: 135
# generated dataset entails: 135
# generated dataset neutral: 133
# generated dataset entails: 133
# generated dataset neutral: 170
# generated dataset entails: 170
# generated dataset neutral: 169
# generated dataset entails: 169
# generated dataset entails: 154
# generated dataset neutral: 154
# generated dataset neutral: 131
# generated dataset entails: 131
# validation dataset for entails: 460
# validation dataset for neutral: 540
# test dataset for neutral: 477
# test dataset for entails: 523
# task1612
# generated dataset 2: 23
# generated dataset 1: 23
# generated dataset 0: 23
# generated dataset 1: 9
# generated dataset 2: 9
# generated dataset 0: 9
# generated dataset 0: 13
# generated dataset 1: 13
# generated dataset 2: 13
# generated dataset 0: 29
# generated dataset 1: 29
# generated dataset 2: 29
# generated dataset 1: 22
# generated dataset 0: 22
# generated dataset 2: 22
# generated dataset 1: 7
# generated dataset 0: 7
# generated dataset 2: 7
# generated dataset 0: 18
# generated dataset 2: 18
# generated dataset 1: 18
# generated dataset 2: 15
# generated dataset 1: 15
# generated dataset 0: 15
# generated dataset 1: 5
# generated dataset 0: 5
# generated dataset 2: 5
# generated dataset 0: 7
# generated dataset 1: 7
# generated dataset 2: 7
# validation dataset for 2: 280
# validation dataset for 1: 266
# validation dataset for 0: 254
# test dataset for 2: 320
# test dataset for 1: 334
# test dataset for 0: 346
# task1615
# generated dataset B_neutral_A: 34
# generated dataset B_entails_A: 34
# generated dataset B_contradicts_A: 34
# generated dataset B_neutral_A: 42
# generated dataset B_entails_A: 42
# generated dataset B_contradicts_A: 42
# generated dataset B_neutral_A: 56
# generated dataset B_contradicts_A: 56
# generated dataset B_entails_A: 56
# generated dataset B_entails_A: 64
# generated dataset B_neutral_A: 64
# generated dataset B_contradicts_A: 64
# generated dataset B_neutral_A: 53
# generated dataset B_contradicts_A: 53
# generated dataset B_entails_A: 53
# generated dataset B_neutral_A: 69
# generated dataset B_contradicts_A: 69
# generated dataset B_entails_A: 69
# generated dataset B_neutral_A: 41
# generated dataset B_contradicts_A: 41
# generated dataset B_entails_A: 41
# generated dataset B_contradicts_A: 69
# generated dataset B_entails_A: 69
# generated dataset B_neutral_A: 69
# generated dataset B_neutral_A: 34
# generated dataset B_entails_A: 34
# generated dataset B_contradicts_A: 34
# generated dataset B_neutral_A: 32
# generated dataset B_contradicts_A: 32
# generated dataset B_entails_A: 32
# validation dataset for B_contradicts_A: 167
# validation dataset for B_neutral_A: 179
# validation dataset for B_entails_A: 154
# test dataset for B_neutral_A: 168
# test dataset for B_contradicts_A: 165
# test dataset for B_entails_A: 167
# task190
# generated dataset E: 14
# generated dataset N: 14
# generated dataset C: 14
# generated dataset N: 2
# generated dataset E: 2
# generated dataset C: 2
# generated dataset E: 4
# generated dataset N: 4
# generated dataset C: 4
# generated dataset E: 1
# generated dataset N: 1
# generated dataset C: 1
# generated dataset N: 4
# generated dataset E: 4
# generated dataset C: 4
# generated dataset E: 5
# generated dataset N: 5
# generated dataset C: 5
# generated dataset E: 3
# generated dataset N: 3
# generated dataset C: 3
# generated dataset N: 5
# generated dataset E: 5
# generated dataset C: 5
# generated dataset E: 6
# generated dataset N: 6
# generated dataset C: 6
# generated dataset E: 50
# generated dataset N: 50
# validation dataset for C: 678
# validation dataset for N: 322
# test dataset for C: 668
# test dataset for N: 332
