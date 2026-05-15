#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import time
import argparse
import logging
import numpy as np
import pyccf5 as slow5


DEFAULT_FEATURE_NAMES = [
    "mean", "std", "median", "min", "max", "range", "iqr",
    "diff_mean", "diff_std", "diff_abs_mean",
    "diff2_mean", "diff2_std",
    "peak_to_peak", "energy"
]


def setup_logger(output_dir: str, log_file: str | None = None):
    os.makedirs(output_dir, exist_ok=True)
    if log_file is None:
        log_file = os.path.join(output_dir, "run.log")

    logger = logging.getLogger("ccf5_chunk")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger, log_file


def normalize_one_chunk(signal: np.ndarray) -> np.ndarray:
    """
    对单个 chunk 做 robust 标准化：
        normalized = (signal - med) / mad

    注意：
    1. 这里 mad 不乘 1.4826，严格按用户给定公式
    2. 若 mad 过小，返回空数组，表示该 chunk 无法可靠标准化
    """
    if signal.size == 0:
        return np.array([], dtype=np.float32)

    med = np.median(signal)
    mad = np.median(np.abs(signal - med))

    if mad < 1e-8:
        return np.array([], dtype=np.float32)

    normalized = (signal - med) / mad
    return normalized.astype(np.float32)


def make_normalized_chunks_with_filter(
    signal: np.ndarray,
    chunk_len: int = 128,
    stride: int = 64,
    value_min: float = -3.0,
    value_max: float = 3.0,
):
    """
    先切 chunk，再对每个 chunk 单独标准化。
    仅保留标准化后所有值都在 [value_min, value_max] 的 chunk。
    否则整个 chunk 剔除。

    返回：
        kept_chunks: (n_kept, chunk_len)
        kept_starts: (n_kept,)
        total_candidates: 候选 chunk 总数
        dropped_chunks: 被剔除的 chunk 数
    """
    n = len(signal)
    if n < chunk_len:
        return (
            np.empty((0, chunk_len), dtype=np.float32),
            np.array([], dtype=np.int64),
            0,
            0,
        )

    starts = np.arange(0, n - chunk_len + 1, stride, dtype=np.int64)

    kept_chunks = []
    kept_starts = []
    total_candidates = len(starts)
    dropped_chunks = 0

    for s in starts:
        chunk = signal[s:s + chunk_len].astype(np.float32)
        chunk_norm = normalize_one_chunk(chunk)

        # mad 过小或标准化失败，直接丢掉该 chunk
        if chunk_norm.size == 0:
            dropped_chunks += 1
            continue

        # 只要有任意值超出范围，就丢掉整个 chunk
        if np.any(chunk_norm < value_min) or np.any(chunk_norm > value_max):
            dropped_chunks += 1
            continue

        kept_chunks.append(chunk_norm)
        kept_starts.append(s)

    if len(kept_chunks) == 0:
        return (
            np.empty((0, chunk_len), dtype=np.float32),
            np.array([], dtype=np.int64),
            total_candidates,
            dropped_chunks,
        )

    return (
        np.stack(kept_chunks, axis=0).astype(np.float32),
        np.array(kept_starts, dtype=np.int64),
        total_candidates,
        dropped_chunks,
    )


def compute_chunk_features(chunks: np.ndarray):
    feature_names = DEFAULT_FEATURE_NAMES

    if len(chunks) == 0:
        return np.empty((0, len(feature_names)), dtype=np.float32), feature_names

    x = chunks
    d1 = np.diff(x, axis=1)
    d2 = np.diff(d1, axis=1)

    mean = np.mean(x, axis=1)
    std = np.std(x, axis=1)
    median = np.median(x, axis=1)
    xmin = np.min(x, axis=1)
    xmax = np.max(x, axis=1)
    xrange = xmax - xmin
    q75 = np.percentile(x, 75, axis=1)
    q25 = np.percentile(x, 25, axis=1)
    iqr = q75 - q25

    diff_mean = np.mean(d1, axis=1)
    diff_std = np.std(d1, axis=1)
    diff_abs_mean = np.mean(np.abs(d1), axis=1)

    diff2_mean = np.mean(d2, axis=1)
    diff2_std = np.std(d2, axis=1)

    peak_to_peak = np.ptp(x, axis=1)
    energy = np.mean(x ** 2, axis=1)

    features = np.stack([
        mean, std, median, xmin, xmax, xrange, iqr,
        diff_mean, diff_std, diff_abs_mean,
        diff2_mean, diff2_std,
        peak_to_peak, energy
    ], axis=1).astype(np.float32)

    return features, feature_names


def safe_get_one_read(s5, read_id):
    """
    单条读取 read，并尽量直接拿 pA 信号。
    关键点：
    1. 不使用 aux='all'，避免 pyccf5 在异常 aux 字段上崩溃
    2. 优先请求 pA=True，使 read['signal'] 直接是 pA 信号
    """
    last_err = None

    for kwargs in (
        {"pA": True},
        {"pA": True, "aux": None},
        {},
    ):
        try:
            return s5.get_read(read_id, **kwargs)
        except Exception as e:
            last_err = e

    raise last_err


def convert_signal_to_pa(read):
    """
    尽可能稳健地返回 pA 信号：
    - 如果 get_read(..., pA=True)，则 read['signal'] 通常已是 pA
    - 如果不是，再尝试旧版字段换算
    """
    if "signal" not in read:
        raise KeyError("signal")

    sig = np.asarray(read["signal"])

    # 情况1：单条 get_read(pA=True) 后，signal 已经是 pA
    if np.issubdtype(sig.dtype, np.floating):
        return sig.astype(np.float32)

    # 情况2：兼容旧字段
    if "lvdsmid" in read and "unit" in read:
        return ((sig - read["lvdsmid"]) * read["unit"]).astype(np.float32)

    if all(k in read for k in ("K", "scale", "offset", "B")):
        return (
            read["K"] * read["scale"] * (sig.astype(np.uint16) + read["offset"]) + read["B"]
        ).astype(np.float32)

    # 情况3：最后兜底，直接转 float
    return sig.astype(np.float32)


def process_one_ccf5(
    ccf5_path: str,
    output_dir: str,
    min_raw_len: int,
    trim_head: int,
    trim_tail: int,
    chunk_len: int,
    stride: int,
    threads: int,
    batchsize: int,
    logger,
    file_index: int,
    total_files: int,
    log_every_reads: int,
    value_min: float,
    value_max: float,
):
    base = os.path.splitext(os.path.basename(ccf5_path))[0]
    out_prefix = os.path.join(output_dir, base)

    all_chunks = []
    all_features = []
    all_read_ids = []
    all_starts = []
    all_trimmed_lengths = []

    total_reads = 0
    kept_reads = 0
    skipped_short_raw = 0
    skipped_after_trim = 0
    skipped_signal_error = 0
    skipped_read_fetch_error = 0
    total_chunks = 0
    total_candidate_chunks = 0
    dropped_chunks_out_of_range = 0

    file_start = time.time()
    logger.info(f"[{file_index}/{total_files}] START file: {ccf5_path}")

    s5 = slow5.Open(ccf5_path, "r", DEBUG=0)
    all_read_ids_in_file = s5.get_read_ids()[0]
    expected_reads = len(all_read_ids_in_file)
    logger.info(f"[{file_index}/{total_files}] {base}: detected {expected_reads} reads")

    bad_read_log = os.path.join(output_dir, f"{base}.bad_read_ids.txt")
    if os.path.exists(bad_read_log):
        os.remove(bad_read_log)

    feature_names = None

    for idx, read_id in enumerate(all_read_ids_in_file, start=1):
        total_reads += 1

        try:
            read = safe_get_one_read(s5, read_id)
        except Exception as e:
            skipped_read_fetch_error += 1
            with open(bad_read_log, "a", encoding="utf-8") as bf:
                bf.write(f"{read_id}\tfetch_error\t{repr(e)}\n")

            if (skipped_read_fetch_error <= 10) or (
                log_every_reads > 0 and total_reads % log_every_reads == 0
            ):
                logger.warning(
                    f"[{file_index}/{total_files}] {base}: failed to fetch read "
                    f"{read_id} | err={repr(e)}"
                )
            continue

        try:
            signal = convert_signal_to_pa(read)
        except Exception as e:
            skipped_signal_error += 1
            with open(bad_read_log, "a", encoding="utf-8") as bf:
                bf.write(f"{read_id}\tsignal_error\t{repr(e)}\n")

            if (skipped_signal_error <= 10) or (
                log_every_reads > 0 and total_reads % log_every_reads == 0
            ):
                logger.warning(
                    f"[{file_index}/{total_files}] {base}: failed to convert signal "
                    f"{read_id} | err={repr(e)}"
                )
            continue

        if len(signal) < min_raw_len:
            skipped_short_raw += 1
            continue

        if len(signal) <= (trim_head + trim_tail):
            skipped_after_trim += 1
            continue

        signal_trimmed = signal[trim_head: len(signal) - trim_tail]

        chunks, starts, total_candidate_chunks_this_read, dropped_chunks_this_read = make_normalized_chunks_with_filter(
            signal_trimmed,
            chunk_len=chunk_len,
            stride=stride,
            value_min=value_min,
            value_max=value_max,
        )

        total_candidate_chunks += total_candidate_chunks_this_read
        dropped_chunks_out_of_range += dropped_chunks_this_read

        if len(chunks) == 0:
            skipped_after_trim += 1
            continue

        features, feature_names = compute_chunk_features(chunks)

        all_chunks.append(chunks)
        all_features.append(features)
        all_read_ids.extend([read_id] * len(chunks))
        all_starts.extend((starts + trim_head).tolist())
        all_trimmed_lengths.extend([len(signal_trimmed)] * len(chunks))

        kept_reads += 1
        total_chunks += len(chunks)

        if log_every_reads > 0 and (total_reads % log_every_reads == 0):
            elapsed = time.time() - file_start
            logger.info(
                f"[{file_index}/{total_files}] {base}: processed {total_reads}/{expected_reads} reads | "
                f"kept={kept_reads} skipped_short={skipped_short_raw} "
                f"skipped_trim={skipped_after_trim} fetch_err={skipped_read_fetch_error} "
                f"signal_err={skipped_signal_error} candidate_chunks={total_candidate_chunks} "
                f"dropped_chunks={dropped_chunks_out_of_range} kept_chunks={total_chunks} "
                f"elapsed={elapsed:.1f}s"
            )

    s5.close()

    if len(all_chunks) > 0:
        all_chunks = np.concatenate(all_chunks, axis=0).astype(np.float32)
        all_features = np.concatenate(all_features, axis=0).astype(np.float32)
    else:
        all_chunks = np.empty((0, chunk_len), dtype=np.float32)
        feat_dim = len(DEFAULT_FEATURE_NAMES) if feature_names is None else len(feature_names)
        all_features = np.empty((0, feat_dim), dtype=np.float32)
        if feature_names is None:
            feature_names = DEFAULT_FEATURE_NAMES

    all_read_ids = np.array(all_read_ids, dtype=object)
    all_starts = np.array(all_starts, dtype=np.int64)
    all_trimmed_lengths = np.array(all_trimmed_lengths, dtype=np.int32)

    logger.info(f"[{file_index}/{total_files}] {base}: saving outputs to {out_prefix}.*")

    np.save(f"{out_prefix}.chunks.npy", all_chunks)
    np.save(f"{out_prefix}.chunk_features.npy", all_features)
    np.savez(
        f"{out_prefix}.chunk_meta.npz",
        read_ids=all_read_ids,
        starts=all_starts,
        trimmed_lengths=all_trimmed_lengths,
        chunk_len=np.array([chunk_len], dtype=np.int32),
        stride=np.array([stride], dtype=np.int32),
        trim_head=np.array([trim_head], dtype=np.int32),
        trim_tail=np.array([trim_tail], dtype=np.int32),
        min_raw_len=np.array([min_raw_len], dtype=np.int32),
        value_min=np.array([value_min], dtype=np.float32),
        value_max=np.array([value_max], dtype=np.float32),
        source_file=np.array([ccf5_path], dtype=object),
    )

    with open(f"{out_prefix}.feature_names.txt", "w", encoding="utf-8") as f:
        for name in feature_names:
            f.write(name + "\n")

    with open(f"{out_prefix}.summary.txt", "w", encoding="utf-8") as f:
        f.write(f"source_file\t{ccf5_path}\n")
        f.write(f"total_reads\t{total_reads}\n")
        f.write(f"kept_reads\t{kept_reads}\n")
        f.write(f"skipped_short_raw\t{skipped_short_raw}\n")
        f.write(f"skipped_after_trim\t{skipped_after_trim}\n")
        f.write(f"skipped_read_fetch_error\t{skipped_read_fetch_error}\n")
        f.write(f"skipped_signal_error\t{skipped_signal_error}\n")
        f.write(f"total_candidate_chunks\t{total_candidate_chunks}\n")
        f.write(f"dropped_chunks_out_of_range\t{dropped_chunks_out_of_range}\n")
        f.write(f"total_chunks\t{total_chunks}\n")
        f.write(f"chunk_len\t{chunk_len}\n")
        f.write(f"stride\t{stride}\n")
        f.write(f"trim_head\t{trim_head}\n")
        f.write(f"trim_tail\t{trim_tail}\n")
        f.write(f"min_raw_len\t{min_raw_len}\n")
        f.write(f"value_min\t{value_min}\n")
        f.write(f"value_max\t{value_max}\n")

    elapsed = time.time() - file_start
    logger.info(
        f"[{file_index}/{total_files}] DONE {base} | reads={total_reads}/{expected_reads} "
        f"kept={kept_reads} skipped_short={skipped_short_raw} skipped_trim={skipped_after_trim} "
        f"fetch_err={skipped_read_fetch_error} signal_err={skipped_signal_error} "
        f"candidate_chunks={total_candidate_chunks} dropped_chunks={dropped_chunks_out_of_range} "
        f"kept_chunks={total_chunks} elapsed={elapsed:.1f}s -> {out_prefix}.*"
    )

    return {
        "total_reads": total_reads,
        "kept_reads": kept_reads,
        "skipped_short_raw": skipped_short_raw,
        "skipped_after_trim": skipped_after_trim,
        "skipped_read_fetch_error": skipped_read_fetch_error,
        "skipped_signal_error": skipped_signal_error,
        "total_candidate_chunks": total_candidate_chunks,
        "dropped_chunks_out_of_range": dropped_chunks_out_of_range,
        "total_chunks": total_chunks,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build chunk-level normalized chunks and chunk features from all .ccf5 files in a directory."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing .ccf5 files")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--pattern", default="*.ccf5", help="Glob pattern for input files")
    parser.add_argument("--min_raw_len", type=int, default=20000, help="Minimum raw signal length")
    parser.add_argument("--trim_head", type=int, default=2000, help="Trim from head")
    parser.add_argument("--trim_tail", type=int, default=2000, help="Trim from tail")
    parser.add_argument("--chunk_len", type=int, default=128, help="Chunk length")
    parser.add_argument("--stride", type=int, default=64, help="Chunk stride")
    parser.add_argument("--threads", type=int, default=1, help="Reserved arg; kept for CLI compatibility")
    parser.add_argument("--batchsize", type=int, default=1, help="Reserved arg; kept for CLI compatibility")
    parser.add_argument("--log_every_reads", type=int, default=500, help="Print progress every N reads")
    parser.add_argument("--log_file", default=None, help="Optional log file path; default is output_dir/run.log")
    parser.add_argument("--value_min", type=float, default=-3.0, help="Min allowed normalized value in a chunk")
    parser.add_argument("--value_max", type=float, default=3.0, help="Max allowed normalized value in a chunk")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger, log_file = setup_logger(args.output_dir, args.log_file)

    logger.info("========== JOB START ==========")
    logger.info(
        "Parameters | "
        f"input_dir={args.input_dir} output_dir={args.output_dir} pattern={args.pattern} "
        f"min_raw_len={args.min_raw_len} trim_head={args.trim_head} trim_tail={args.trim_tail} "
        f"chunk_len={args.chunk_len} stride={args.stride} threads={args.threads} "
        f"batchsize={args.batchsize} log_every_reads={args.log_every_reads} "
        f"value_min={args.value_min} value_max={args.value_max} log_file={log_file}"
    )

    ccf5_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if len(ccf5_files) == 0:
        raise FileNotFoundError(f"No files matched: {os.path.join(args.input_dir, args.pattern)}")

    logger.info(f"Found {len(ccf5_files)} files")

    job_start = time.time()
    grand_total_reads = 0
    grand_kept_reads = 0
    grand_skipped_short = 0
    grand_skipped_trim = 0
    grand_fetch_err = 0
    grand_signal_err = 0
    grand_total_candidate_chunks = 0
    grand_dropped_chunks = 0
    grand_total_chunks = 0

    for idx, ccf5_path in enumerate(ccf5_files, start=1):
        stats = process_one_ccf5(
            ccf5_path=ccf5_path,
            output_dir=args.output_dir,
            min_raw_len=args.min_raw_len,
            trim_head=args.trim_head,
            trim_tail=args.trim_tail,
            chunk_len=args.chunk_len,
            stride=args.stride,
            threads=args.threads,
            batchsize=args.batchsize,
            logger=logger,
            file_index=idx,
            total_files=len(ccf5_files),
            log_every_reads=args.log_every_reads,
            value_min=args.value_min,
            value_max=args.value_max,
        )
        grand_total_reads += stats["total_reads"]
        grand_kept_reads += stats["kept_reads"]
        grand_skipped_short += stats["skipped_short_raw"]
        grand_skipped_trim += stats["skipped_after_trim"]
        grand_fetch_err += stats["skipped_read_fetch_error"]
        grand_signal_err += stats["skipped_signal_error"]
        grand_total_candidate_chunks += stats["total_candidate_chunks"]
        grand_dropped_chunks += stats["dropped_chunks_out_of_range"]
        grand_total_chunks += stats["total_chunks"]

    total_elapsed = time.time() - job_start
    logger.info(
        "ALL DONE | "
        f"files={len(ccf5_files)} total_reads={grand_total_reads} kept_reads={grand_kept_reads} "
        f"skipped_short={grand_skipped_short} skipped_trim={grand_skipped_trim} "
        f"fetch_err={grand_fetch_err} signal_err={grand_signal_err} "
        f"candidate_chunks={grand_total_candidate_chunks} dropped_chunks={grand_dropped_chunks} "
        f"kept_chunks={grand_total_chunks} elapsed={total_elapsed:.1f}s"
    )
    logger.info("========== JOB END ==========")


if __name__ == "__main__":
    main()