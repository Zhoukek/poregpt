import numpy as np


N = 537111

# # 随机打乱 index
perm = np.random.permutation(N)

chunk_npy="/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-teacher_model-strill1/recon/chunks_all.npy"
ref_npy="/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-teacher_model-strill1/recon/references_all.npy"
ref_len="/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-teacher_model-strill1/recon/reference_lengths_all.npy"


chunks=np.load(chunk_npy)
references=np.load(ref_npy)
reference_lengths=np.load(ref_len)

chunks = chunks[perm]
references = references[perm]
reference_lengths = reference_lengths[perm]

np.save("/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-teacher_model-strill1/shuffle_npy/chunks.npy", chunks)
np.save("/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-teacher_model-strill1/shuffle_npy/references.npy", references)
np.save("/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-teacher_model-strill1/shuffle_npy/reference_lengths.npy", reference_lengths)

