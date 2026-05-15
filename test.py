import torch

ckpt_path = "/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-haojieshen-model-type26-cnn_type13_teacher_model_distill0.1_VQ_64k_lemon/output_150m_ctx1280-lr5e4-vqe_teacher-codebook-aware-test-v0.1/steps/latest-unsharded/model.pt"
sd = torch.load(ckpt_path, map_location="cpu")

for k in sd.keys():
    if "code_q_proj" in k or "code_k_proj" in k:
        print(k, sd[k].shape)