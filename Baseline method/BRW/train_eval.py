import numpy as np
import json
from data_loader import load_data, build_graph
from graph_model import generate_embeddings

def recommend_top_n(model, mashup_ids, api_ids, N=20):
    scores = {}
    for m in mashup_ids:
        if m not in model.wv: continue
        m_vec = model.wv[m]
        scores[m] = sorted([(a, np.dot(m_vec, model.wv[a])) for a in api_ids if a in model.wv], key=lambda x: -x[1])[:N]
    return scores

def evaluate_recall(recommendations, ground_truth, ks=(3, 5, 10, 20)):
    recalls = {k: [] for k in ks}
    for m_id, recs in recommendations.items():
        pred_apis = [a for a, _ in recs]
        true_apis = ground_truth.get(m_id, set())
        if not true_apis:
            continue
        for k in ks:
            hits = len(set(pred_apis[:k]) & true_apis)
            recalls[k].append(hits / len(true_apis))
    return {f"Recall@{k}": np.mean(recalls[k]) if recalls[k] else 0.0 for k in ks}

def run_train_test_split(mashup_path="data/mashups.json", api_path="data/apis.json", topN=20):
    mashups, apis = load_data(mashup_path, api_path)
    mashup_ids = [m['id'] for m in mashups]
    mashup_dict = {m['id']: m for m in mashups}

    # ✅ 使用前70%训练，后30%测试
    split_point = int(len(mashup_ids) * 0.7)
    train_ids = set(mashup_ids[:split_point])
    test_ids = set(mashup_ids[split_point:])

    print(f"Total mashups: {len(mashup_ids)}, Train: {len(train_ids)}, Test: {len(test_ids)}")

    # 构建图时仅用训练集连接
    G, mashup_nodes, api_nodes = build_graph(mashups, apis, test_ids=test_ids)

    model = generate_embeddings(G, dimensions=128, walk_length=20, num_walks=80, phi=0.4, omega=0.6)

    test_mashup_nodes = [f"m_{mid}" for mid in test_ids]
    recs = recommend_top_n(model, test_mashup_nodes, api_nodes, N=topN)

    # 构建 ground truth
    gt = {}
    for mid in test_ids:
        mkey = f"m_{mid}"
        if mkey in G:
            true_apis = {nbr for nbr in G.neighbors(mkey) if G.edges[mkey, nbr]['type'] == 'call'}
            gt[mkey] = true_apis

    # 保存 JSONL 推荐结果
    output_path = f"BRW.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for mkey, ranked_apis in recs.items():
            mid = int(mkey.split('_')[1])
            rec_api_ids = [int(a.split('_')[1]) for a, _ in ranked_apis]
            removed_api_ids = [int(nbr.split('_')[1]) for nbr in gt.get(mkey, [])]
            result = {
                "mashup_id": mid,
                "remove_apis": removed_api_ids,
                "recommend_api": rec_api_ids
            }
            f.write(json.dumps(result) + '\n')

# if __name__ == "__main__":
#     run_train_test_split("data/new_mashups.json", "data/new_apis.json")

def main():
    run_train_test_split("data/new_mashups.json", "data/new_apis.json")

if __name__ == '__main__':
    from memory_profiler import memory_usage

    mem_usage = memory_usage(main, interval=0.1)
    print(f"Peak memory: {max(mem_usage):.2f} MB")
