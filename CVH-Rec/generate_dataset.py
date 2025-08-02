import json
import os
import h5py
import numpy as np
from sentence_transformers import SentenceTransformer

# 加载数据
with open('Original Dataset/mashups.json', 'r', encoding='utf-8') as f:
    mashups = json.load(f)

with open('Original Dataset/apis.json', 'r', encoding='utf-8') as f:
    apis = json.load(f)

# 构造API的id到tags映射
api_id_to_tags = {}
all_api_tags = set()
for api in apis:
    api_id = api['id']
    tags_str = api['details'].get('tags', '')
    tags = [t.strip() for t in tags_str.split(',') if t.strip()]
    api_id_to_tags[api_id] = tags
    all_api_tags.update(tags)

# 构造mashup的id到categories映射
mashup_id_to_tags = {}
all_mashup_tags = set()
for m in mashups:
    mid = m['id']
    cats = [c.strip() for c in m['categories'].split(',') if c.strip()]
    mashup_id_to_tags[mid] = cats
    all_mashup_tags.update(cats)

# 合并所有tag，生成统一的tag id映射
all_tags = all_api_tags.union(all_mashup_tags)
tag2id = {tag: i for i, tag in enumerate(sorted(all_tags))}

# 初始化SentenceTransformer模型
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def save_vectors_h5(filename, id_to_text):
    with h5py.File(filename, 'w') as f:
        for id_, text in id_to_text.items():
            vec = model.encode(text)
            f.create_dataset(str(id_), data=vec.astype(np.float32))

os.makedirs('data', exist_ok=True)
# 生成所有tag的向量（统一tag）
save_vectors_h5('data/tag_vectors.h5', {tid: tag for tag, tid in tag2id.items()})

# 生成API description向量
api_id_to_desc = {api['id']: api['details']['description'] for api in apis}
save_vectors_h5('data/API_vectors.h5', api_id_to_desc)

# 生成mashup description向量
mashup_id_to_desc = {m['id']: m['description'] for m in mashups}
save_vectors_h5('data/vectors.h5', mashup_id_to_desc)

# ========= 生成70%训练 + 20%测试 + 10%验证文件 =========
N = len(mashups)
train_end = round(N * 0.7)
test_end = round(N * 0.9)   # 70% + 20%
print(test_end)

train_set = mashups[:train_end]        # 前 70%
test_set = mashups[train_end:test_end] # 中间 20%
val_set = mashups[test_end:]           # 后 10%

split_dir = 'dataset/fold'
os.makedirs(split_dir, exist_ok=True)

# 写 TE.csv（训练集）
with open(os.path.join(split_dir, 'TE.csv'), 'w', encoding='utf-8') as f:
    for m in train_set:
        line = [str(m['id'])] + [str(api_id) for api_id in m.get('api_info', [])]
        f.write(' '.join(line) + '\n')

# 写 RS.csv（测试集）
with open(os.path.join(split_dir, 'RS.csv'), 'w', encoding='utf-8') as f:
    for m in test_set:
        line = [str(m['id'])] + [str(api_id) for api_id in m.get('api_info', [])]
        f.write(' '.join(line) + '\n')

# 写 VA.csv（验证集）
with open(os.path.join(split_dir, 'VA.csv'), 'w', encoding='utf-8') as f:
    for m in val_set:
        line = [str(m['id'])] + [str(api_id) for api_id in m.get('api_info', [])]
        f.write(' '.join(line) + '\n')

# 写统一tag映射
with open(os.path.join(split_dir, 'tag_mapping.csv'), 'w', encoding='utf-8') as f:
    for tag, tid in tag2id.items():
        f.write(f'{tid}, {tag}\n')

# 写 api_tags.csv
with open(os.path.join(split_dir, 'api_tags.csv'), 'w', encoding='utf-8') as f:
    for api_id, tags in api_id_to_tags.items():
        tag_ids = [str(tag2id[t]) for t in tags if t in tag2id]
        f.write(f'{api_id} ' + ' '.join(tag_ids) + '\n')

# 写 mashup_tags.csv（只写训练集中的 mashup）
with open(os.path.join(split_dir, 'mashup_tags.csv'), 'w', encoding='utf-8') as f:
    for m in train_set:
        tags = mashup_id_to_tags.get(m['id'], [])
        tag_ids = [str(tag2id[t]) for t in tags if t in tag2id]
        f.write(f"{m['id']} " + ' '.join(tag_ids) + '\n')