import numpy as np


# N = 537111

# # 随机打乱 index
# perm = np.random.permutation(N)


# chunk_npy="/mnt/zzbnew/dnadata/data/balanced/250F701901011/human_min0_max2_read96655/basecall/validation/chunk_basecall/chunks.npy"
# ref_npy="/mnt/zzbnew/dnadata/data/balanced/250F701901011/human_min0_max2_read96655/basecall/validation/chunk_basecall/references.npy"
# ref_len="/mnt/zzbnew/dnadata/data/balanced/250F701901011/human_min0_max2_read96655/basecall/validation/chunk_basecall/reference_lengths.npy"


# chunks=np.load(chunk_npy)
# references=np.load(ref_npy)
# reference_lengths=np.load(ref_len)

# chunks = chunks[perm]
# references = references[perm]
# reference_lengths = reference_lengths[perm]

# np.save("data/raw_signal/shuffle_npy/chunks.npy", chunks)
# np.save("data/raw_signal/shuffle_npy/references.npy", references)
# np.save("data/raw_signal/shuffle_npy/reference_lengths.npy", reference_lengths)


# chunk_npy="data/apple_signal/chunks.npy"
# ref_npy="data/apple_signal/references.npy"
# ref_len="data/apple_signal/reference_lengths.npy"


# chunks=np.load(chunk_npy)
# references=np.load(ref_npy)
# reference_lengths=np.load(ref_len)

# chunks = chunks[perm]
# references = references[perm]
# reference_lengths = reference_lengths[perm]

# np.save("data/apple_signal//shuffle_npy/chunks.npy", chunks)
# np.save("data/apple_signal//shuffle_npy/references.npy", references)
# np.save("data/apple_signal//shuffle_npy/reference_lengths.npy", reference_lengths)


# chunk_npy="data/stone_signal/chunks.npy"
# ref_npy="data/stone_signal/references.npy"
# ref_len="data/stone_signal/reference_lengths.npy"

# chunks=np.load(chunk_npy)
# references=np.load(ref_npy)
# reference_lengths=np.load(ref_len)

# chunks = chunks[perm]
# references = references[perm]
# reference_lengths = reference_lengths[perm]

# np.save("data/stone_signal//shuffle_npy/chunks.npy", chunks)
# np.save("data/stone_signal//shuffle_npy/references.npy", references)
# np.save("data/stone_signal//shuffle_npy/reference_lengths.npy", reference_lengths)


N = 527544

# # 随机打乱 index
perm = np.random.permutation(N)


# chunk_npy="data/HF_20m_DNA_VQE16K_CNN01_V20260121/validation_chunks_all.npy"
# ref_npy="data/HF_20m_DNA_VQE16K_CNN01_V20260121/validation_references_all.npy"
# ref_len="data/HF_20m_DNA_VQE16K_CNN01_V20260121/validation_reference_lengths_all.npy"


# chunks=np.load(chunk_npy)
# references=np.load(ref_npy)
# reference_lengths=np.load(ref_len)

# chunks = chunks[perm]
# references = references[perm]
# reference_lengths = reference_lengths[perm]

# np.save("data/HF_20m_DNA_VQE16K_CNN01_V20260121//shuffle_npy/chunks.npy", chunks)
# np.save("data/HF_20m_DNA_VQE16K_CNN01_V20260121//shuffle_npy/references.npy", references)
# np.save("data/HF_20m_DNA_VQE16K_CNN01_V20260121//shuffle_npy/reference_lengths.npy", reference_lengths)





chunk_npy="/mnt/zzbnew/rnamodel/shenhaojie/data/test-zhoukexuan-model-type25-cnn_type2-teacher_model_distill_1/chunks_all.npy"
ref_npy="/mnt/zzbnew/rnamodel/shenhaojie/data/test-zhoukexuan-model-type25-cnn_type2-teacher_model_distill_1/references_all.npy"
ref_len="/mnt/zzbnew/rnamodel/shenhaojie/data/test-zhoukexuan-model-type25-cnn_type2-teacher_model_distill_1/reference_lengths_all.npy"


chunks=np.load(chunk_npy)
references=np.load(ref_npy)
reference_lengths=np.load(ref_len)

chunks = chunks[perm]
references = references[perm]
reference_lengths = reference_lengths[perm]

np.save("/mnt/zzbnew/rnamodel/shenhaojie/poregpt/DNA_basecall/tokenize/data/test-zhoukexuan-model-type25-cnn_type2-teacher_model_distill_1/shuffle_npy/chunks.npy", chunks)
np.save("/mnt/zzbnew/rnamodel/shenhaojie/poregpt/DNA_basecall/tokenize/data/test-zhoukexuan-model-type25-cnn_type2-teacher_model_distill_1/shuffle_npy/references.npy", references)
np.save("/mnt/zzbnew/rnamodel/shenhaojie/poregpt/DNA_basecall/tokenize/data/test-zhoukexuan-model-type25-cnn_type2-teacher_model_distill_1/shuffle_npy/reference_lengths.npy", reference_lengths)

# chunk_npy="data/lemon_signal//chunks.npy"
# ref_npy="data/lemon_signal/references.npy"
# ref_len="data/lemon_signal/reference_lengths.npy"


# chunks=np.load(chunk_npy)
# references=np.load(ref_npy)
# reference_lengths=np.load(ref_len)

# chunks = chunks[perm]
# references = references[perm]
# reference_lengths = reference_lengths[perm]

# np.save("data/lemon_signal//shuffle_npy/chunks.npy", chunks)
# np.save("data/lemon_signal//shuffle_npy/references.npy", references)
# np.save("data/lemon_signal//shuffle_npy/reference_lengths.npy", reference_lengths)





