import argparse
import gzip
import json
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm

from ont_fast5_api.fast5_interface import get_fast5_file
from poregpt.tokenizers import VQETokenizer
from poregpt.utils import nanopore_process_signal
import bonito

def main():
    # 1. 设置命令行参数
    parser = argparse.ArgumentParser(description="Process jsonl.gz and convert to npy")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing .jsonl.gz files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save .npy files")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()

    DEVICE = "cuda"
    TOKEN_BATCH_SIZE = 100

    tokenizer = VQETokenizer(
        model_ckpt=args.model_ckpt,
        device=DEVICE,
        token_batch_size=TOKEN_BATCH_SIZE
    )

    # 2. 从参数获取路径
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
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

if __name__ == "__main__":
    main()