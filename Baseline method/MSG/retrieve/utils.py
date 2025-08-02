import json
import random

random_seed=1
random.seed(random_seed)

# def get_indices():
#     with open('./data/mashup_name.json', 'r') as file:
#         dataset = json.load(file)
#     split_num = int(len(dataset) / 10)
#     test_idx = dataset[:split_num]
#     train_idx = dataset[split_num:]
#     print("len(train_idx), len(test_idx)----------------------", len(train_idx), len(test_idx))
#
#
#
#     train_apis = set()
#     oov_api = set()
#     print('oov {}'.format(len(oov_api)))
#     return train_idx, test_idx, oov_api

def get_indices():
    train_ratio = 0.7  # 前 70% 训练
    with open('./data/mashup_name.json', 'r') as file:
        dataset = json.load(file)

    total_size = len(dataset)
    split_point = int(total_size * train_ratio)

    train_idx = dataset[:split_point]
    test_idx = dataset[split_point:]

    print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    train_apis = set()
    oov_api = set()
    print(f'oov {len(oov_api)}')

    return train_idx, test_idx, oov_api

