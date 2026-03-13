
if false; then
    python3 scripts/step07_count_code_dist2.py  \
	/mnt/nas_syy/dataset/huada_rna_80G/liuh/data/rna/npy_chunks_32_5/fast5_q85/cluster_8192/cluster_texts \
	--workers 32 --output codebook_hist_kms_ruaq95_w32s05.png
fi

if false; then
    python3 scripts/step07_count_code_dist2.py  \
	/mnt/nas_syy/dataset/huada_rna_80G/liuh/data/rna/npy_chunks_32_5/fast5_full/chunk_cluster_8192_32_5/2b_5r_centroids_8192/cluster_text \
	--workers 32 --output codebook_hist_kms_ruaq95_w32s05.png
fi

if true; then
    python3 scripts/step07_count_code_dist2.py  \
	/mnt/nas_syy/dataset/huada_rna_80G/huada_rna_Q85_vqm_cnntype0_2loss_pass15/spoch15000_fast5_jsonlgz \
	--workers 32 --output codebook_hist.png
fi

