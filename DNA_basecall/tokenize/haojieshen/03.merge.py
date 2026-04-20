import numpy as np
from pathlib import Path
import pandas as pd

data_dir = Path("/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-teacher_model-strill1/recon")

# 找文件
chunk_files = sorted(data_dir.glob("*_chunks.npy"))
ref_files = sorted(data_dir.glob("*_references.npy"))
len_files = sorted(data_dir.glob("*_reference_lengths.npy"))
summary_data=pd.read_csv("/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-teacher_model-strill1/recon/processing_summary.csv")
file_list=[i.split(".")[0] for i  in summary_data['fast5_name']]

print("chunks files:", len(chunk_files))
print("references files:", len(ref_files))
print("length files:", len(len_files))



chunk_files=["/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-teacher_model-strill1/recon/{}_chunks.npy".format(i) for i in file_list]
ref_files=["/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-teacher_model-strill1/recon/{}_references.npy".format(i) for i in file_list]
len_files=["/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-teacher_model-strill1/recon/{}_reference_lengths.npy".format(i) for i in file_list]

def merge_npy(files):
    arrays = []
    for f in files:
        arrays.append(np.load(f))
    return np.concatenate(arrays, axis=0)


# 合并
chunks = merge_npy(chunk_files)
references = merge_npy(ref_files)
reference_lengths = merge_npy(len_files)

print("merged shapes:")
print("chunks:", chunks.shape)
print("references:", references.shape)
print("reference_lengths:", reference_lengths.shape)


# 保存
np.save(data_dir / "chunks_all.npy", chunks)
np.save(data_dir / "references_all.npy", references)
np.save(data_dir / "reference_lengths_all.npy", reference_lengths)

print("merge finished")




