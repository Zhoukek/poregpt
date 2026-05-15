from poregpt.tokenizers import VQETokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


tokenizer_path = "/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-haojieshen-model-type26-cnn_type13_teacher_model_distill0.1_VQ_64k_lemon/encoder"

tokenizer = VQETokenizer(model_ckpt=tokenizer_path)


# print(tokenizer.vq)
# print(dir(tokenizer))
codebook = tokenizer._get_codebook_embed().detach().cpu().numpy()

print(codebook.shape)


# 采样5000-10000个点
sample_size = min(65536, codebook.shape[0])
indices = np.random.choice(codebook.shape[0], sample_size, replace=False)
codebook_sample = codebook[indices]

# t-SNE降维
tsne = TSNE(n_components=2, random_state=42, perplexity=30, 
            n_iter=1000, verbose=1)
codebook_2d = tsne.fit_transform(codebook_sample)

# 绘图
plt.figure(figsize=(12, 10))
scatter = plt.scatter(codebook_2d[:, 0], codebook_2d[:, 1], 
                      c=indices, cmap='plasma', s=5, alpha=0.6)
plt.colorbar(scatter, label='Code Index', pad=0.02)
plt.title(f't-SNE Visualization (sampled {sample_size}/{codebook.shape[0]} codes)')
plt.grid(True, alpha=0.3)
plt.savefig('/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/DNA_modification/wangxue/workflow/result/codebook_tsne.png', dpi=300)
plt.show()