input_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result/
out_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result/models/model2/


python -m script.mod_v1.train_model2_cluster \
  --native_residual ${input_dir}/infer/model1_out/native_model1_residual.npy \
  --canonical_residual ${input_dir}/models/model1/canonical_model1_residual.npy \
  --out_dir ${out_dir} \
  --n_states 64 \
  --pca_dim 30 \
  --max_train_native 200000
