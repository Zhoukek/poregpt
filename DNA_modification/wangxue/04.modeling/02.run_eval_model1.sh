input_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result/models/model1
python -m script.mod_v1.evaluate_model1 \
  --input ${input_dir}/canonical_model1_input.npy \
  --recon ${input_dir}/canonical_model1_recon.npy \
  --residual ${input_dir}/canonical_model1_residual.npy \
  --train_history  ${input_dir}/train_history.json \
  --latent ${input_dir}/canonical_model1_latent.npy \
  --mse ${input_dir}/canonical_model1_recon_mse.npy \
  --source_index ${input_dir}/canonical_model1_source_index.npy \
  --source_map ${input_dir}/source_map.json \
  --out_dir ${input_dir}/eval
