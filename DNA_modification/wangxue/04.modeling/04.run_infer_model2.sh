input_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result/infer/model1_out
model_ckpt=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result/models/model2
out_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result/infer/model2_out



python -m script.mod_v1.infer_model2 \
  --residual ${input_dir}/synthetic_model1_residual.npy \
  --model2_pca  ${model_ckpt}/model2_pca.joblib \
  --model2_kmeans ${model_ckpt}/model2_kmeans.joblib \
  --out_dir ${out_dir} \
  --prefix synthetic_model2
