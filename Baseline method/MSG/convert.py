import json

input_file = "decode_greedy29.jsonl"
output_file = "MSR.json"

records = []

# 读取并转换每条记录
with open(input_file, "r", encoding="utf-8") as fin:
    for line in fin:
        data = json.loads(line)
        mashup_id = int(data["mashup_name"])
        recommend_api = [int(api) for api in data["predict_apis"] if api != "<pad>"]
        records.append({
            "mashup_id": mashup_id,
            "recommend_api": recommend_api
        })

# 按 mashup_id 排序
records.sort(key=lambda x: x["mashup_id"])

# 写入结果
with open(output_file, "w", encoding="utf-8") as fout:
    for record in records:
        fout.write(json.dumps(record) + "\n")
