import json
import math

if __name__ == '__main__':
    Ns = [3, 5, 10, 20]

    # 加载 API ID -> 名称 映射
    with open("data/new_apis.json", "r", encoding="utf-8") as f:
        api_data = json.load(f)
        api_id_to_title = {item["id"]: item["url"] for item in api_data}
        all_api_ids = set(api_id_to_title.keys())

    # --- 统计调用历史：mashup.json中前80%记录 ---
    with open(f"data/new_mashups.json", "r", encoding="utf-8") as f:
        mashup_data = json.load(f)

    top_80_count = int(len(mashup_data) * 0.8)

    hit_counts = {api_id: 0 for api_id in all_api_ids}
    total_interactions = 0

    for i, mashup in enumerate(mashup_data[:top_80_count]):
        api_list = mashup.get("api_info", [])
        for api_id in api_list:
            if api_id in hit_counts:
                hit_counts[api_id] += 1
                total_interactions += 1

    # 其余代码保持不变
    # 加载 ground truth

    recommendations = {}
    ground_truth = {}
    method = 'BRW'
    with open(f"{method}.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            mashup_id = data["mashup_id"]
            recommend_api = data["recommend_api"]
            remove_apis = data["remove_apis"]  # 这是 ground truth
            recommendations[mashup_id] = recommend_api
            ground_truth[mashup_id] = remove_apis

    print(total_interactions)
    print(hit_counts.get(1297, 0) / total_interactions)

    print(f"The performance of {method}  on specified mashup_ids are as follows:")
    for N in Ns:
        precision_list = []
        recall_list = []
        map_list = []
        ndcg_list = []
        recommended_api_set = set()
        pb_list = []

        for i, (mashup_id, true_apis) in enumerate(ground_truth.items()):
            if i >= 1433:
            # if i >= 1233:
                break
            recommended_apis = recommendations.get(mashup_id, [])[:N]
            if not true_apis:
                continue

            true_set = set(true_apis)
            hit_count = len(set(recommended_apis) & true_set)

            recommended_api_set.update(recommended_apis)

            precision = hit_count / N
            precision_list.append(precision)

            recall = hit_count / len(true_apis)
            recall_list.append(recall)

            cor_list = [1.0 if recommended_apis[i] in true_set else 0.0
                        for i in range(len(recommended_apis))]
            sum_cor_list = sum(cor_list)
            if sum_cor_list == 0:
                map_score = 0.0
            else:
                summary = sum(
                    sum(cor_list[:i + 1]) / (i + 1) * cor_list[i]
                    for i in range(len(cor_list))
                )
                map_score = summary / sum_cor_list
            map_list.append(map_score)

            dcg = sum(
                1 / math.log2(i + 2) if i < len(recommended_apis) and recommended_apis[i] in true_set else 0
                for i in range(N)
            )
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(true_apis), N)))
            ndcg = dcg / idcg if idcg != 0 else 0
            ndcg_list.append(ndcg)

            # 计算 PB@N
            if total_interactions > 0:
                pb_value = sum(hit_counts.get(api_id, 0) / total_interactions for api_id in recommended_apis)
            else:
                pb_value = 0
            pb_list.append(pb_value)

        average_precision = sum(precision_list) / len(precision_list) if precision_list else 0
        average_recall = sum(recall_list) / len(recall_list) if recall_list else 0
        average_map = sum(map_list) / len(map_list) if map_list else 0
        average_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0
        coverage = len(recommended_api_set) / len(all_api_ids) if all_api_ids else 0
        average_pb = sum(pb_list) / len(pb_list) if pb_list else 0

        # print(f"N={N} -> Precision: {average_precision:.4f}, Recall: {average_recall:.4f}, "
        #       f"MAP: {average_map:.4f}, NDCG: {average_ndcg:.4f}, Coverage: {coverage:.4f}, PB@N: {average_pb:.4f}")
        print(f"N={N} -> & {average_precision:.4f} & {average_recall:.4f}"
              f"& {average_map:.4f} & {average_ndcg:.4f}")
