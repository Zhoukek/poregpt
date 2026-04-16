from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
import json
import gzip
import json
import numpy as np
import re
from poregpt.tokenizers import VQETokenizer
from poregpt.utils import nanopore_process_signal
from tqdm import tqdm
import bonito
from tqdm import tqdm
import gzip
import json
import re
import numpy as np
from pathlib import Path


MODEL_CKPT = "/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/vqe_workflow/step02_train_vqe_model/test-zhoukexuan-model-type25-cnn_type13-baseline/models/porepgt_vqe_tokenizer.final.pth"

DEVICE = "cuda"
TOKEN_BATCH_SIZE = 100

tokenizer = VQETokenizer(
    model_ckpt=MODEL_CKPT,
    device=DEVICE,
    token_batch_size=TOKEN_BATCH_SIZE
)

print(tokenizer)


input_dir = Path("/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-haojieshen-model-type27-cnn_type13-teacher_model/basecall")
output_dir = Path("/mnt/zzbnew/rnamodel/zhoukexuan/data/test-zhoukexuan-model-type25-cnn_type13-baseline")
output_dir.mkdir(parents=True, exist_ok=True)

pattern = re.compile(r"\d+")

target_signal_len = 6000
target_base_len = 762

jsonl_files = sorted(input_dir.glob("*.jsonl.gz"))

print(f"找到 {len(jsonl_files)} 个 jsonl.gz 文件")

for file_idx, jsonl_filepath in enumerate(jsonl_files, 1):
    print(f"\n[{file_idx}/{len(jsonl_files)}] 开始处理: {jsonl_filepath.name}")

    base_list = []
    signal_list = []
    seq_len_list = []

    with gzip.open(jsonl_filepath, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"processing {jsonl_filepath.name}", unit="read"):
            data = json.loads(line)

            # ---------- signal ----------
            text = data["text"]
            raw_tokens = [int(x) for x in pattern.findall(text)]

            signal = np.array(tokenizer.decode_token_ids(raw_tokens))

            if len(signal) < target_signal_len:
                signal = np.pad(signal, (0, target_signal_len - len(signal)), mode="constant")
            else:
                signal = signal[:target_signal_len]

            # ---------- base ----------
            bases = data["bases"]
            seq_len = len(bases)

            base = np.fromiter(bases, dtype=int)

            if len(base) < target_base_len:
                base = np.pad(base, (0, target_base_len - len(base)), mode="constant")
            else:
                base = base[:target_base_len]

            # ---------- collect ----------
            base_list.append(base)
            signal_list.append(signal)
            seq_len_list.append(seq_len)

    # ---------- convert to ndarray ----------
    base_array = np.asarray(base_list, dtype=np.int8)         # (n, 762)
    signal_array = np.asarray(signal_list, dtype=np.float32)  # (n, 6000)
    seq_len_array = np.asarray(seq_len_list, dtype=np.int32)  # (n,)

    print(f"{jsonl_filepath.name} 处理完成")
    print(f"  base_array shape: {base_array.shape}")
    print(f"  signal_array shape: {signal_array.shape}")
    print(f"  seq_len_array shape: {seq_len_array.shape}")

    # ---------- save ----------
    stem = jsonl_filepath.name.replace(".jsonl.gz", "")

    signal_out = output_dir / f"{stem}_chunks.npy"
    base_out = output_dir / f"{stem}_references.npy"
    seq_len_out = output_dir / f"{stem}_reference_lengths.npy"

    np.save(signal_out, signal_array)
    np.save(base_out, base_array)
    np.save(seq_len_out, seq_len_array)

    print("  已保存:")
    print(f"    {signal_out}")
    print(f"    {base_out}")
    print(f"    {seq_len_out}")

print("\n全部处理完成")

import numpy as np
from pathlib import Path
import pandas as pd

data_dir = Path("/mnt/zzbnew/rnamodel/zhoukexuan/data/test-zhoukexuan-model-type25-cnn_type13-baseline")

# 找文件
chunk_files = sorted(data_dir.glob("*_chunks.npy"))
ref_files = sorted(data_dir.glob("*_references.npy"))
len_files = sorted(data_dir.glob("*_reference_lengths.npy"))
summary_data=pd.read_csv("./data/lemon_signal/processing_summary.csv")
file_list=[i.split(".")[0] for i  in summary_data['fast5_name']]

print("chunks files:", len(chunk_files))
print("references files:", len(ref_files))
print("length files:", len(len_files))



chunk_files=["/mnt/zzbnew/rnamodel/zhoukexuan/data/test-zhoukexuan-model-type25-cnn_type13-baseline/{}_chunks.npy".format(i) for i in file_list]
ref_files=["/mnt/zzbnew/rnamodel/zhoukexuan/data/test-zhoukexuan-model-type25-cnn_type13-baseline/{}_references.npy".format(i) for i in file_list]
len_files=["/mnt/zzbnew/rnamodel/zhoukexuan/data/test-zhoukexuan-model-type25-cnn_type13-baseline/{}_reference_lengths.npy".format(i) for i in file_list]

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




