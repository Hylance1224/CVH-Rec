import json

# 读取推荐结果
def load_recommendations(file_path):
    recommendations = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            mashup_id = data["mashup_id"]
            recommend_api = data["recommend_api"]
            recommendations[mashup_id] = recommend_api
    return recommendations

# 计算Recall@3
def compute_recall_at_k(ground_truth, recommendations, k=3):
    recall_per_mashup = {}
    for mashup_id, true_apis in ground_truth.items():
        true_set = set(true_apis)
        rec_apis = recommendations.get(mashup_id, [])[:k]
        hit_count = len(set(rec_apis) & true_set)
        recall = hit_count / len(true_set) if true_set else 0.0
        recall_per_mashup[mashup_id] = recall
    return recall_per_mashup

if __name__ == '__main__':
    # 加载 ground truth
    fold = 5
    ground_truth = {}
    with open(f"dataset/fold_{fold}/RS.csv", "r") as f:
        for line in f:
            parts = line.strip().split()
            mashup_id = int(parts[0])
            apis = list(map(int, parts[1:]))
            ground_truth[mashup_id] = apis

    # 加载两个推荐结果
    r2api_rec = load_recommendations(f"output/R2API_5_0.6_0.6_0.8_0.8_fold_{fold}.json")
    cl_rec = load_recommendations(f"output/CL_5_0.6_0.6_0.6_0.6_fold_{fold}.json")

    # 分别计算Recall@3
    r2api_recall = compute_recall_at_k(ground_truth, r2api_rec, k=3)
    cl_recall = compute_recall_at_k(ground_truth, cl_rec, k=3)

    # 对比并输出CL Recall@3 > R2API Recall@3的mashup_id
    better_mashups_3 = []
    for mashup_id in ground_truth.keys():
        if cl_recall.get(mashup_id, 0) >= r2api_recall.get(mashup_id, 0):
            better_mashups_3.append(mashup_id)

    print("CL Recall@3 > R2API Recall@3 的 mashup_id:")
    print(better_mashups_3)
    print(len(better_mashups_3))

    #-===================================================

    # 分别计算Recall@5
    r2api_recall = compute_recall_at_k(ground_truth, r2api_rec, k=5)
    cl_recall = compute_recall_at_k(ground_truth, cl_rec, k=5)

    # 对比并输出CL Recall@5 > R2API Recall@5的mashup_id
    better_mashups_5 = []
    for mashup_id in ground_truth.keys():
        if cl_recall.get(mashup_id, 0) >= r2api_recall.get(mashup_id, 0):
            better_mashups_5.append(mashup_id)

    print("CL Recall@5 > R2API Recall@5 的 mashup_id:")
    print(better_mashups_5)
    print(len(better_mashups_5))


    intersection = list(set(better_mashups_3) & set(better_mashups_5))
    print(intersection)
    print(len(intersection))



