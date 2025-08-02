import json
import math
from utility.parser import parse_args
args = parse_args()

if __name__ == '__main__':
    Ns = [3, 5, 10, 20]

    # 加载 ground truth
    fold = args.dataset
    ground_truth = {}
    with open(f"dataset/{fold}/RS.csv", "r") as f:
        for line in f:
            parts = line.strip().split()
            mashup_id = int(parts[0])
            apis = list(map(int, parts[1:]))
            ground_truth[mashup_id] = apis


    method = 'CL_DNN'
    recommendations = {}
    with open(
            f"output/{method}_{args.alpha1}_{args.alpha2}_{args.alpha3}_{args.alpha4}_{fold}.json",
            "r") as f:
        for line in f:
            data = json.loads(line)
            mashup_id = data["mashup_id"]
            recommend_api = data["recommend_api"]
            recommendations[mashup_id] = recommend_api

    print(f"The performance of CVH-Rec are as follows:")
    for N in Ns:
        precision_list = []
        recall_list = []
        map_list = []
        ndcg_list = []
        recommended_api_set = set()
        pb_list = []

        for i, (mashup_id, true_apis) in enumerate(ground_truth.items()):
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

        average_precision = sum(precision_list) / len(precision_list) if precision_list else 0
        average_recall = sum(recall_list) / len(recall_list) if recall_list else 0
        average_map = sum(map_list) / len(map_list) if map_list else 0
        average_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0

        print(f"N={N} -> Precision: {average_precision:.4f}, Recall: {average_recall:.4f}, "
              f"MAP: {average_map:.4f}, NDCG: {average_ndcg:.4f}")
