
jsonl_path=/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB09.original_adjust.jsonl.gz


python -m script.make_kmer_signal_table \
  --data-jsonl $jsonl_path \
  --out-tsv ./result/kmer_signal_table.raw_span.tsv \
  --span-field base_sample_spans_rel \
  --base-offset 4 \
  --kmer-size 5