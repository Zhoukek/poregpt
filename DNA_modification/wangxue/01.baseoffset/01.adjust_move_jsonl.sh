#!/bin/bash 

pen=50
kernel=linear
max_shift=100


jsonl_path_in=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/03.token/result_0331/LB07_1/signal_none.jsonl
jsonl_path_out=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/03.token/result_0331/LB07_1/signal_none.adjusted.jsonl

# jsonl_path_in=/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB06.original.jsonl.gz
# jsonl_path_out=/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB06.original_adjust.jsonl.gz


python -m script.adjust_boundaies_jsonl \
    --input ${jsonl_path_in} \
    --output ${jsonl_path_out} \
    --kernel ${kernel} \
    --pen ${pen} \
    --mode left \
    --max-shift ${max_shift} \
    --keep-edge-boundaries-fixed \



python -m script.span_offset_eval \
  --data-jsonl ${jsonl_path_out} \
  --out-dir ./result/span_${pen}_${kernel}_${max_shift}_stats_4 \
  --span-field base_sample_spans_rel \
  --offsets -4 \
  --limit 10000 \
  --prefix ${pen}_${kernel}_${max_shift}_stats_4



python -m script.span_offset_eval \
  --data-jsonl ${jsonl_path_out} \
  --out-dir ./result/span_${pen}_${kernel}_${max_shift}_stats_37 \
  --span-field base_sample_spans_rel \
  --offsets -3.7 \
  --limit 10000 \
  --prefix ${pen}_${kernel}_${max_shift}_stats_37


python -m script.span_offset_eval \
  --data-jsonl ${jsonl_path_out} \
  --out-dir ./result/span_${pen}_${kernel}_${max_shift}_stats_auto \
  --span-field base_sample_spans_rel_adj \
  --offsets -4 \
  --limit 10000 \
  --prefix ${pen}_${kernel}_${max_shift}_stats_auto