#!/bin/bash

# Tokenizer生成脚本
# 功能：根据指定的K值生成tokenizer.json文件

# 设置K值（码本大小）- 请在此处修改K值
K=4096

# 设置输出文件路径 - 请在此处修改输出文件名
OUTPUT="tokenizer-4k.json"

# 打印执行信息
echo "生成tokenizer，K=$K，输出到 $OUTPUT"

# 执行Python生成脚本
python generate_tokenizer.py --K $K --output $OUTPUT

# 打印完成信息
echo "完成!"
