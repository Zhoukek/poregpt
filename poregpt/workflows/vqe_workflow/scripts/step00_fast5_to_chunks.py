#!/usr/bin/env python3
"""
Preprocess Nanopore FAST5 files using the Fast5Dir class directly.

This script replaces the high-level workflow function with direct instantiation
of Fast5Dir and calls to_chunks() with advanced chunking options:
- Multi-phase head cutting (for stride alignment)
- Tail fallback chunking

Usage:
    python run_preprocess.py --fast5_dir ./fast5 --output_dir ./chunks
"""

import argparse
from poregpt.tokenizers.utils import Fast5Dir  # 注意：根据你的实际包结构调整导入路径


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Nanopore FAST5 signals into normalized, phase-aware chunks for LLM/VQ training."
    )
    parser.add_argument(
        "--fast5_dir", type=str, required=True,
        help="Path to directory containing .fast5 files."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save output .npy chunk files."
    )
    parser.add_argument(
        "--window_size", type=int, default=12000,
        help="Length of each signal chunk in samples (default: 12000)."
    )
    parser.add_argument(
        "--stride", type=int, default=11900,
        help="Stride between chunks in samples (default: 11900, i.e., 100-sample overlap)."
    )
    parser.add_argument(
        "--do_normalize", action="store_true",
        help="Enable median-MAD normalization (default: enabled)."
    )
    parser.add_argument(
        "--no_normalize", action="store_false", dest="do_normalize",
        help="Disable normalization."
    )
    parser.set_defaults(do_normalize=True)

    parser.add_argument(
        "--do_medianfilter", action="store_true",
        help="Apply median filter with kernel=5 (default: disabled)."
    )
    parser.add_argument(
        "--do_lowpassfilter", action="store_true",
        help="Apply Butterworth low-pass filter (default: disabled)."
    )
    parser.add_argument(
        "--default_fs", type=int, default=5000,
        help="Default sampling frequency (Hz) if not found in metadata (default: 5000)."
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1,
        help="Number of parallel jobs (-1 means all CPU cores; default: -1)."
    )

    # === 新增参数：多头裁剪与末尾兜底 ===
    parser.add_argument(
        "--cut_head_all", type=int, default=4,
        help="Maximum number of leading samples to cut (inclusive). "
             "Used to generate multiple alignment phases. Default: 11 (covers stride=12)."
    )
    parser.add_argument(
        "--cut_head_step", type=int, default=1,
        help="Step size for head cutting. Default: 1 (i.e., cuts at 0,1,2,...,11)."
    )
    parser.add_argument(
        "--tail_threshold", type=int, default=6000,
        help="Minimum tail length (in samples) to trigger a fallback chunk from the end. Default: 6000."
    )
    parser.add_argument(
        "--signal_min_value", type=int, default=-1000,
        help="Number of parallel jobs (-1 means all CPU cores; default: -1)."
    )
    parser.add_argument(
        "--signal_max_value", type=int, default= 1000,
        help="Number of parallel jobs (-1 means all CPU cores; default: -1)."
    )
    parser.add_argument(
        "--normal_min_value", type=float, default=-9.0,
        help="."
    )
    parser.add_argument(
        "--normal_max_value", type=float, default=9.0,
        help="Number of parallel jobs (-1 means all CPU cores; default: -1)."
    )

    args = parser.parse_args()

    # Instantiate Fast5Dir directly
    fast5_processor = Fast5Dir(
        fast5_dir=args.fast5_dir,
        default_fs=args.default_fs
    )

    # Perform chunking with advanced options
    fast5_processor.to_chunks(
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        do_normalize=args.do_normalize,
        do_medianfilter=args.do_medianfilter,
        do_lowpassfilter=args.do_lowpassfilter,
        cut_head_all=args.cut_head_all,
        cut_head_step=args.cut_head_step,
        tail_threshold=args.tail_threshold,
        n_jobs=args.n_jobs,
        signal_min_value = args.signal_min_value,
        signal_max_value = args.signal_max_value,
        normal_min_value = args.normal_min_value,
        normal_max_value = args.normal_max_value,
    )

if __name__ == "__main__":
    main()
