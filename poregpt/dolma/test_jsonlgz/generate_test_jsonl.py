# generate_bwav_jsonl.py

import json
import random

# -------------------------------
# 配置参数
# -------------------------------
OUTPUT_FILE = "demo.jsonl"      # 输出文件名
NUM_LINES = 1000                     # 生成多少行
MIN_TOKENS_PER_LINE = 9              # 每行最少几个 bwav token
MAX_TOKENS_PER_LINE = 200             # 每行最多几个 bwav token
VOCAB_SIZE = 8192                    # token ID 范围：0 ~ 8191

# -------------------------------
# 生成函数
# -------------------------------
def generate_bwav_text(num_tokens):
    """生成包含 num_tokens 个 <|bwav:i|> 的字符串"""
    tokens = [f"<|bwav:{random.randint(0, VOCAB_SIZE - 1)}|>" for _ in range(num_tokens)]
    return "".join(tokens)

def main():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i in range(1, NUM_LINES + 1):
            num_tokens = random.randint(MIN_TOKENS_PER_LINE, MAX_TOKENS_PER_LINE)
            text = generate_bwav_text(num_tokens)
            line = {"id": str(i), "text": text}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    
    print(f"✅ 已生成 {NUM_LINES} 行数据到 {OUTPUT_FILE}")

# -------------------------------
# 运行
# -------------------------------
if __name__ == "__main__":
    main()
