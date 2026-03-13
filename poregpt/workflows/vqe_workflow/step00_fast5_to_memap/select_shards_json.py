import json
import random

# 读取原始shards.json文件
with open('shards.json', 'r') as file:
    shards_data = json.load(file)

# 设置你想要抽取的比例（例如0.5代表50%）
sample_ratio = 0.5

# 根据设定的比例随机选择shards
selected_shards = random.sample(shards_data['shards'], int(len(shards_data['shards']) * sample_ratio))

# 计算新选取的shards中的样本总数
total_selected_samples = sum([shard['num_samples'] for shard in selected_shards])

# 创建新的shards数据结构
new_shards_data = {
    "total_samples": total_selected_samples,
    "chunk_size": shards_data['chunk_size'],
    "dtype": shards_data['dtype'],
    "shards": selected_shards
}

# 将新选取的shards保存到shards_new.json
with open('shards_new.json', 'w') as file:
    json.dump(new_shards_data, file, indent=2)

print(f"已根据{sample_ratio*100}%的比例从原始shards中选择了新的shards，并保存为shards_new.json")
