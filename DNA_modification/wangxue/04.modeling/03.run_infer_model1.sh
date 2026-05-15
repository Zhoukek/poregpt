input_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result_mod2/corpus/native_chunks
model_ckpt=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result/models/model1/
out_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result/infer



python -m script.mod_v1.infer_model1 \
 --chunks ${input_dir}/250F701586012_0_1_0.chunks.npy ${input_dir}/250F701586012_0_1_1.chunks.npy ${input_dir}/250F701586012_0_1_2.chunks.npy \
 --model_ckpt ${model_ckpt}/model1_best.pt \
 --out_dir ${out_dir}/model1_out \
 --batch_size 512 \
 --prefix native_model1 



input_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result_mod2/corpus/synthetic_chunks
model_ckpt=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result/models/model1/
out_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result/infer
# python -m script.mod_v1.infer_model1 \
#  --chunks ${input_dir}/chunks.npy \
#  --model_ckpt ${model_ckpt}/model1_best.pt \
#  --out_dir ${out_dir}/model1_out \
#  --batch_size 512 \
#  --prefix synthetic_model1
