python -m script.mod_v1.visualize_synthetic_read_residual \
  --jsonl /mnt/zzbnew/rnamodel/wangxue/DNA_modification/token/result_0331/LB06_2/signal_none.jsonl \
  --chunk_meta /mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result_mod2/corpus/synthetic_chunks/chunk_meta.npz \
  --residual /mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result/infer/model1_out/synthetic_model1_residual.npy \
  --state_id /mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result/infer/model2_out/synthetic_model2_state_id.npy \
  --read_id 250F600084012_1_202_674_11920089_12557 \
  --mod_bases 9,28,47,66,85,104,123 \
  --out_dir synthetic_read_vis \
  --n_examples 6
 
