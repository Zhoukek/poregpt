#!/bin/bash
jsonl_path_out=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/03.token/result_0331/LB07_1/signal_none.adjusted.jsonl

python -m script.plot_base_boxplot_raw_vs_adj \
  --data-jsonl ${jsonl_path_out}\
  --out-dir result/base_boxplot \
  --limit 1000