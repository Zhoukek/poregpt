#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对 jsonl/jsonl.gz 中每条 read 的 base_sample_spans_rel 进行基于 physical boundaries 的局部保守校正，
并将结果写入新字段：base_sample_spans_rel_adj。

支持三种校正模式：

1. right
   - 每个中间内部边界只允许向右搜索 physical boundary

2. left
   - 每个中间内部边界只允许向左搜索 physical boundary

3. both-best
   - 每个中间内部边界同时在左、右两个方向搜索 physical boundary
   - 将原始边界本身也作为候选
   - 根据局部信号左右均值差选择最优边界
   - 若最优候选相对原始边界没有达到 min-improvement，则保留原始边界

输入 jsonl 每行至少包含：
- signal
- base_sample_spans_rel

输出 jsonl 每行新增：
- base_sample_spans_rel_adj

支持：
- input.jsonl
- input.jsonl.gz
- output.jsonl
- output.jsonl.gz

示例：

python adjust_boundaries_jsonl_best_gz.py \
    --input input.jsonl.gz \
    --output output.left.jsonl.gz \
    --mode left \
    --kernel rbf \
    --pen 1.0 \
    --max-shift 20 \
    --keep-edge-boundaries-fixed

python adjust_boundaries_jsonl_best_gz.py \
    --input input.jsonl.gz \
    --output output.both_best.jsonl.gz \
    --mode both-best \
    --kernel rbf \
    --pen 1.0 \
    --max-shift 20 \
    --min-improvement 0.0 \
    --keep-edge-boundaries-fixed
"""

from __future__ import annotations

import os
import json
import gzip
import argparse
from bisect import bisect_left, bisect_right
from typing import Any

import numpy as np
import ruptures as rpt


def smart_open(path: str, mode: str = "rt", encoding: str = "utf-8"):
    """
    自动支持普通文本文件和 gzip 文件。

    读取：
        smart_open("input.jsonl", "rt")
        smart_open("input.jsonl.gz", "rt")

    写出：
        smart_open("output.jsonl", "wt")
        smart_open("output.jsonl.gz", "wt")
    """
    path = str(path)

    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding=encoding)

    return open(path, mode, encoding=encoding)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "基于 ruptures physical boundaries，对 base_sample_spans_rel 进行 "
            "left/right/both-best 局部保守校正，并输出新 jsonl/jsonl.gz。"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="输入 jsonl 或 jsonl.gz 文件路径",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 jsonl 或 jsonl.gz 文件路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both-best",
        choices=["right", "left", "both-best"],
        help=(
            "边界校正模式："
            "right=只向右搜索；"
            "left=只向左搜索；"
            "both-best=左右都搜索并按局部信号评分选最优。默认 both-best"
        ),
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "cosine"],
        help="ruptures KernelCPD 的 kernel 类型，默认 rbf",
    )
    parser.add_argument(
        "--pen",
        type=float,
        default=1.0,
        help="ruptures predict 的 penalty 参数，默认 1.0",
    )
    parser.add_argument(
        "--max-shift",
        type=int,
        default=20,
        help="边界最大允许移动的 signal sample 数，默认 20",
    )
    parser.add_argument(
        "--min-seg-len",
        type=int,
        default=1,
        help="校正后相邻边界的最小间隔，默认 1",
    )
    parser.add_argument(
        "--min-side-len",
        type=int,
        default=1,
        help="both-best 模式下，候选边界左右两侧用于评分的最小 signal 长度，默认 1",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help=(
            "both-best 模式下，最佳候选 score 至少比原始边界 score 高多少才替换。"
            "默认 0.0"
        ),
    )
    parser.add_argument(
        "--keep-edge-boundaries-fixed",
        action="store_true",
        help="是否固定首尾内部边界不校正。建议开启。",
    )
    parser.add_argument(
        "--max-reads",
        type=int,
        default=None,
        help="最多处理多少条 read，默认处理全部",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=100,
        help="每处理多少条 read 打印一次进度，默认 100",
    )
    parser.add_argument(
        "--write-failed",
        action="store_true",
        help="如果某条 read 处理失败，是否把原 record 写入输出文件。默认不写。",
    )

    return parser.parse_args()


def validate_signal(signal: Any) -> np.ndarray:
    if not isinstance(signal, (list, tuple, np.ndarray)):
        raise ValueError("signal 必须是 list/tuple/ndarray")

    if len(signal) == 0:
        raise ValueError("signal 不能为空")

    try:
        x = np.asarray(signal, dtype=float)
    except Exception as e:
        raise ValueError(f"signal 无法转换为 float 数组: {e}")

    if x.ndim != 1:
        x = x.reshape(-1)

    if len(x) == 0:
        raise ValueError("signal 转换后为空")

    if not np.all(np.isfinite(x)):
        raise ValueError("signal 中包含 NaN 或 Inf")

    return x


def validate_spans(spans: Any, signal_len: int) -> list[list[int]]:
    if not isinstance(spans, list):
        raise ValueError("base_sample_spans_rel 必须是 list")

    if len(spans) == 0:
        raise ValueError("base_sample_spans_rel 不能为空")

    parsed: list[list[int]] = []

    for i, span in enumerate(spans):
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError(
                f"第 {i} 个 span 非法，应为长度为 2 的 list/tuple，实际={span}"
            )

        s, e = span

        try:
            s = int(s)
            e = int(e)
        except Exception:
            raise ValueError(f"第 {i} 个 span 无法转为 int: {span}")

        if not (0 <= s < e <= signal_len):
            raise ValueError(
                f"第 {i} 个 span 越界或长度非法: [{s}, {e}], signal_len={signal_len}"
            )

        parsed.append([s, e])

    if parsed[0][0] != 0:
        raise ValueError(f"首个 span 必须从 0 开始，实际为 {parsed[0]}")

    if parsed[-1][1] != signal_len:
        raise ValueError(
            f"最后一个 span 必须以 signal_len 结束，实际为 {parsed[-1]}, "
            f"signal_len={signal_len}"
        )

    for i in range(1, len(parsed)):
        prev_e = parsed[i - 1][1]
        cur_s = parsed[i][0]

        if prev_e != cur_s:
            raise ValueError(
                f"spans 不连续：第 {i - 1} 个 span 结束为 {prev_e}，"
                f"第 {i} 个 span 起始为 {cur_s}"
            )

    return parsed


def internal_boundaries_from_spans(
    spans: list[list[int]],
    signal_len: int,
) -> list[int]:
    if len(spans) <= 1:
        return []

    bkps = [int(span[1]) for span in spans[:-1]]

    for i, b in enumerate(bkps):
        if not (1 <= b < signal_len):
            raise ValueError(f"内部边界非法：第 {i} 个边界={b}, signal_len={signal_len}")

    return bkps


def internal_boundaries_to_spans(
    boundaries: list[int],
    signal_len: int,
) -> list[list[int]]:
    spans: list[list[int]] = []
    prev = 0

    for b in boundaries:
        b = int(b)
        spans.append([prev, b])
        prev = b

    spans.append([prev, int(signal_len)])

    return spans


def unique_sorted_internal_boundaries(
    boundaries: list[int],
    signal_len: int,
) -> list[int]:
    return sorted(set(int(b) for b in boundaries if 1 <= int(b) < signal_len))


def detect_physical_boundaries(
    signal: np.ndarray,
    kernel: str = "rbf",
    pen: float = 1.0,
) -> list[int]:
    x = np.asarray(signal, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    signal_len = x.shape[0]

    if signal_len <= 1:
        return []

    algo = rpt.KernelCPD(kernel=kernel).fit(x)
    bkps = algo.predict(pen=pen)

    if len(bkps) > 0 and int(bkps[-1]) == signal_len:
        bkps = bkps[:-1]

    return unique_sorted_internal_boundaries(bkps, signal_len)


def enforce_monotonic_boundaries(
    boundaries: list[int],
    signal_len: int,
    min_seg_len: int = 1,
) -> list[int]:
    if len(boundaries) == 0:
        return []

    if min_seg_len < 1:
        raise ValueError(f"min_seg_len 必须 >= 1，实际为 {min_seg_len}")

    boundaries = [int(b) for b in boundaries]
    n = len(boundaries)

    min_required = (n + 1) * min_seg_len
    if signal_len < min_required:
        raise ValueError(
            f"无法满足最小分段长度约束: signal_len={signal_len}, "
            f"内部边界数={n}, min_seg_len={min_seg_len}, "
            f"理论最小需求={min_required}"
        )

    out: list[int] = []
    prev = 0

    for i, b in enumerate(boundaries):
        remain_internal = n - i - 1

        low = prev + min_seg_len
        high = signal_len - (remain_internal + 1) * min_seg_len

        if low > high:
            raise ValueError(
                f"无法构造合法单调边界: i={i}, low={low}, high={high}, "
                f"signal_len={signal_len}, min_seg_len={min_seg_len}"
            )

        b_new = min(max(int(b), low), high)
        out.append(int(b_new))
        prev = int(b_new)

    return out


def find_candidates_in_closed_interval(
    sorted_bkps: list[int],
    left_closed: int,
    right_closed: int,
) -> list[int]:
    """
    在闭区间 [left_closed, right_closed] 内找候选 internal boundaries。
    """
    if left_closed > right_closed:
        return []

    left_idx = bisect_left(sorted_bkps, left_closed)
    right_idx = bisect_right(sorted_bkps, right_closed)

    return sorted_bkps[left_idx:right_idx]


def choose_nearest_candidate(
    candidates: list[int],
    original_b: int,
) -> int | None:
    """
    单方向模式下的默认策略：
    选离原始边界最近的候选；若并列，选较小者。
    """
    if not candidates:
        return None

    original_b = int(original_b)

    return min(candidates, key=lambda p: (abs(int(p) - original_b), int(p)))


def get_directional_search_interval(
    model_bkps: list[int],
    i: int,
    signal_len: int,
    direction: str,
    max_shift: int,
) -> tuple[int, int] | None:
    """
    为第 i 个内部边界生成单方向、受限范围的搜索区间 [low, high]。

    对中间边界：
    - right: [b_i, min(b_i + max_shift, b_{i+1} - 1)]
    - left:  [max(b_{i-1} + 1, b_i - max_shift), b_i]
    """
    n = len(model_bkps)
    b = int(model_bkps[i])

    if i <= 0 or i >= n - 1:
        return None

    left_neighbor = int(model_bkps[i - 1])
    right_neighbor = int(model_bkps[i + 1])

    if not (left_neighbor < b < right_neighbor):
        raise ValueError(
            f"原始边界关系非法: left={left_neighbor}, b={b}, right={right_neighbor}"
        )

    if direction == "right":
        low = b
        high = min(b + max_shift, right_neighbor - 1)
    elif direction == "left":
        low = max(left_neighbor + 1, b - max_shift)
        high = b
    else:
        raise ValueError(f"未知 direction: {direction}")

    low = max(1, int(low))
    high = min(signal_len - 1, int(high))

    if low > high:
        return None

    return low, high


def get_bidirectional_search_intervals(
    model_bkps: list[int],
    i: int,
    signal_len: int,
    max_shift: int,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """
    为第 i 个内部边界生成 left 和 right 两个受限搜索区间。

    left:
        [max(b_{i-1} + 1, b_i - max_shift), b_i]

    right:
        [b_i, min(b_i + max_shift, b_{i+1} - 1)]
    """
    n = len(model_bkps)
    b = int(model_bkps[i])

    if i <= 0 or i >= n - 1:
        return None, None

    left_neighbor = int(model_bkps[i - 1])
    right_neighbor = int(model_bkps[i + 1])

    if not (left_neighbor < b < right_neighbor):
        raise ValueError(
            f"原始边界关系非法: left={left_neighbor}, b={b}, right={right_neighbor}"
        )

    left_low = max(left_neighbor + 1, b - max_shift)
    left_high = b

    right_low = b
    right_high = min(b + max_shift, right_neighbor - 1)

    left_low = max(1, int(left_low))
    left_high = min(signal_len - 1, int(left_high))

    right_low = max(1, int(right_low))
    right_high = min(signal_len - 1, int(right_high))

    left_interval = (left_low, left_high) if left_low <= left_high else None
    right_interval = (right_low, right_high) if right_low <= right_high else None

    return left_interval, right_interval


def score_boundary_by_local_mean_diff(
    signal: np.ndarray,
    candidate_b: int,
    left_limit: int,
    right_limit: int,
    min_side_len: int = 1,
) -> float | None:
    """
    用候选边界 candidate_b 将局部信号分成左右两段，
    计算左右均值差作为 boundary score。

    score 越大，说明该候选边界越像真实 signal transition。
    """
    candidate_b = int(candidate_b)
    left_limit = int(left_limit)
    right_limit = int(right_limit)

    if min_side_len < 1:
        raise ValueError(f"min_side_len 必须 >= 1，实际为 {min_side_len}")

    if not (left_limit < candidate_b < right_limit):
        return None

    left = np.asarray(signal[left_limit:candidate_b], dtype=float)
    right = np.asarray(signal[candidate_b:right_limit], dtype=float)

    if len(left) < min_side_len or len(right) < min_side_len:
        return None

    score = abs(float(np.mean(left)) - float(np.mean(right)))

    if not np.isfinite(score):
        return None

    return score


def init_empty_stats(n_physical_bkps: int = 0) -> dict[str, int | float]:
    return {
        "n_model_bkps": 0,
        "n_physical_bkps": int(n_physical_bkps),
        "n_mode_changed": 0,
        "n_directional_changed": 0,
        "n_bidirectional_changed": 0,
        "n_enforced_changed": 0,
        "n_total_changed": 0,
        "n_fixed_edges": 0,
        "n_no_interval_kept": 0,
        "n_no_candidate_kept": 0,
        "n_intervals_evaluated": 0,
        "n_candidates_total": 0,
        "n_left_selected": 0,
        "n_right_selected": 0,
        "n_original_selected": 0,
    }


def correct_boundaries_directional(
    model_bkps: list[int],
    physical_bkps: list[int],
    signal_len: int,
    direction: str = "right",
    max_shift: int = 20,
    min_seg_len: int = 1,
    keep_edge_boundaries_fixed: bool = True,
) -> tuple[list[int], dict[str, int | float]]:
    """
    单方向、受限范围的边界修正。
    """
    model_bkps = [int(x) for x in model_bkps]
    physical_bkps = unique_sorted_internal_boundaries(physical_bkps, signal_len)

    if direction not in {"right", "left"}:
        raise ValueError(f"direction 必须是 right 或 left，实际为 {direction}")

    if max_shift < 0:
        raise ValueError(f"max_shift 必须 >= 0，实际为 {max_shift}")

    for i, b in enumerate(model_bkps):
        if not (1 <= b < signal_len):
            raise ValueError(f"model 内部边界非法: idx={i}, b={b}, signal_len={signal_len}")

    if any(model_bkps[i] >= model_bkps[i + 1] for i in range(len(model_bkps) - 1)):
        raise ValueError("model_bkps 不是严格递增的")

    n = len(model_bkps)

    if n == 0:
        return [], init_empty_stats(n_physical_bkps=len(physical_bkps))

    corrected_pre_enforce: list[int] = []

    n_directional_changed = 0
    n_fixed_edges = 0
    n_no_interval_kept = 0
    n_no_candidate_kept = 0
    n_intervals_evaluated = 0
    n_candidates_total = 0
    n_left_selected = 0
    n_right_selected = 0
    n_original_selected = 0

    for i, b in enumerate(model_bkps):
        b = int(b)

        if keep_edge_boundaries_fixed and (i == 0 or i == n - 1):
            corrected_pre_enforce.append(b)
            n_fixed_edges += 1
            n_original_selected += 1
            continue

        if i == 0 or i == n - 1:
            corrected_pre_enforce.append(b)
            n_original_selected += 1
            continue

        interval = get_directional_search_interval(
            model_bkps=model_bkps,
            i=i,
            signal_len=signal_len,
            direction=direction,
            max_shift=max_shift,
        )

        if interval is None:
            corrected_pre_enforce.append(b)
            n_no_interval_kept += 1
            n_original_selected += 1
            continue

        low, high = interval
        n_intervals_evaluated += 1

        candidates = find_candidates_in_closed_interval(
            sorted_bkps=physical_bkps,
            left_closed=low,
            right_closed=high,
        )
        n_candidates_total += len(candidates)

        best = choose_nearest_candidate(candidates, original_b=b)

        if best is None:
            corrected_pre_enforce.append(b)
            n_no_candidate_kept += 1
            n_original_selected += 1
        else:
            best = int(best)
            corrected_pre_enforce.append(best)

            if best != b:
                n_directional_changed += 1

                if best < b:
                    n_left_selected += 1
                elif best > b:
                    n_right_selected += 1
            else:
                n_original_selected += 1

    corrected = enforce_monotonic_boundaries(
        corrected_pre_enforce,
        signal_len=signal_len,
        min_seg_len=min_seg_len,
    )

    n_enforced_changed = sum(
        1 for a, b in zip(corrected_pre_enforce, corrected) if int(a) != int(b)
    )
    n_total_changed = sum(
        1 for a, b in zip(model_bkps, corrected) if int(a) != int(b)
    )

    detail_stats = {
        "n_model_bkps": len(model_bkps),
        "n_physical_bkps": len(physical_bkps),
        "n_mode_changed": n_directional_changed,
        "n_directional_changed": n_directional_changed,
        "n_bidirectional_changed": 0,
        "n_enforced_changed": n_enforced_changed,
        "n_total_changed": n_total_changed,
        "n_fixed_edges": n_fixed_edges,
        "n_no_interval_kept": n_no_interval_kept,
        "n_no_candidate_kept": n_no_candidate_kept,
        "n_intervals_evaluated": n_intervals_evaluated,
        "n_candidates_total": n_candidates_total,
        "n_left_selected": n_left_selected,
        "n_right_selected": n_right_selected,
        "n_original_selected": n_original_selected,
    }

    return corrected, detail_stats


def correct_boundaries_bidirectional_best(
    model_bkps: list[int],
    physical_bkps: list[int],
    signal: np.ndarray,
    signal_len: int,
    max_shift: int = 20,
    min_seg_len: int = 1,
    keep_edge_boundaries_fixed: bool = True,
    min_side_len: int = 1,
    min_improvement: float = 0.0,
) -> tuple[list[int], dict[str, int | float]]:
    """
    双方向候选搜索 + 局部 signal score 选择最优边界。
    """
    model_bkps = [int(x) for x in model_bkps]
    physical_bkps = unique_sorted_internal_boundaries(physical_bkps, signal_len)
    signal = np.asarray(signal, dtype=float)

    if max_shift < 0:
        raise ValueError(f"max_shift 必须 >= 0，实际为 {max_shift}")

    if min_side_len < 1:
        raise ValueError(f"min_side_len 必须 >= 1，实际为 {min_side_len}")

    for i, b in enumerate(model_bkps):
        if not (1 <= b < signal_len):
            raise ValueError(f"model 内部边界非法: idx={i}, b={b}, signal_len={signal_len}")

    if any(model_bkps[i] >= model_bkps[i + 1] for i in range(len(model_bkps) - 1)):
        raise ValueError("model_bkps 不是严格递增的")

    n = len(model_bkps)

    if n == 0:
        return [], init_empty_stats(n_physical_bkps=len(physical_bkps))

    corrected_pre_enforce: list[int] = []

    n_bidirectional_changed = 0
    n_fixed_edges = 0
    n_no_interval_kept = 0
    n_no_candidate_kept = 0
    n_intervals_evaluated = 0
    n_candidates_total = 0
    n_left_selected = 0
    n_right_selected = 0
    n_original_selected = 0

    for i, b in enumerate(model_bkps):
        b = int(b)

        if keep_edge_boundaries_fixed and (i == 0 or i == n - 1):
            corrected_pre_enforce.append(b)
            n_fixed_edges += 1
            n_original_selected += 1
            continue

        if i == 0 or i == n - 1:
            corrected_pre_enforce.append(b)
            n_original_selected += 1
            continue

        left_interval, right_interval = get_bidirectional_search_intervals(
            model_bkps=model_bkps,
            i=i,
            signal_len=signal_len,
            max_shift=max_shift,
        )

        if left_interval is None and right_interval is None:
            corrected_pre_enforce.append(b)
            n_no_interval_kept += 1
            n_original_selected += 1
            continue

        left_neighbor = int(model_bkps[i - 1])
        right_neighbor = int(model_bkps[i + 1])

        candidates: set[int] = {b}

        if left_interval is not None:
            low, high = left_interval
            left_candidates = find_candidates_in_closed_interval(
                sorted_bkps=physical_bkps,
                left_closed=low,
                right_closed=high,
            )
            candidates.update(int(x) for x in left_candidates)
            n_intervals_evaluated += 1

        if right_interval is not None:
            low, high = right_interval
            right_candidates = find_candidates_in_closed_interval(
                sorted_bkps=physical_bkps,
                left_closed=low,
                right_closed=high,
            )
            candidates.update(int(x) for x in right_candidates)
            n_intervals_evaluated += 1

        n_candidates_total += len(candidates)

        if len(candidates) == 1 and b in candidates:
            corrected_pre_enforce.append(b)
            n_no_candidate_kept += 1
            n_original_selected += 1
            continue

        original_score = score_boundary_by_local_mean_diff(
            signal=signal,
            candidate_b=b,
            left_limit=left_neighbor,
            right_limit=right_neighbor,
            min_side_len=min_side_len,
        )

        scored_candidates: list[tuple[int, float]] = []

        for c in candidates:
            c = int(c)
            score = score_boundary_by_local_mean_diff(
                signal=signal,
                candidate_b=c,
                left_limit=left_neighbor,
                right_limit=right_neighbor,
                min_side_len=min_side_len,
            )

            if score is None:
                continue

            scored_candidates.append((c, float(score)))

        if not scored_candidates:
            corrected_pre_enforce.append(b)
            n_no_candidate_kept += 1
            n_original_selected += 1
            continue

        best_b, best_score = max(
            scored_candidates,
            key=lambda x: (x[1], -abs(x[0] - b), -x[0]),
        )

        if original_score is None:
            original_score = -np.inf

        if best_b != b and best_score >= float(original_score) + float(min_improvement):
            corrected_pre_enforce.append(int(best_b))
            n_bidirectional_changed += 1

            if best_b < b:
                n_left_selected += 1
            elif best_b > b:
                n_right_selected += 1
            else:
                n_original_selected += 1
        else:
            corrected_pre_enforce.append(b)
            n_original_selected += 1

    corrected = enforce_monotonic_boundaries(
        corrected_pre_enforce,
        signal_len=signal_len,
        min_seg_len=min_seg_len,
    )

    n_enforced_changed = sum(
        1 for a, b2 in zip(corrected_pre_enforce, corrected) if int(a) != int(b2)
    )
    n_total_changed = sum(
        1 for a, b2 in zip(model_bkps, corrected) if int(a) != int(b2)
    )

    detail_stats = {
        "n_model_bkps": len(model_bkps),
        "n_physical_bkps": len(physical_bkps),
        "n_mode_changed": n_bidirectional_changed,
        "n_directional_changed": 0,
        "n_bidirectional_changed": n_bidirectional_changed,
        "n_enforced_changed": n_enforced_changed,
        "n_total_changed": n_total_changed,
        "n_fixed_edges": n_fixed_edges,
        "n_no_interval_kept": n_no_interval_kept,
        "n_no_candidate_kept": n_no_candidate_kept,
        "n_intervals_evaluated": n_intervals_evaluated,
        "n_candidates_total": n_candidates_total,
        "n_left_selected": n_left_selected,
        "n_right_selected": n_right_selected,
        "n_original_selected": n_original_selected,
    }

    return corrected, detail_stats


def adjust_one_record(
    record: dict[str, Any],
    mode: str = "both-best",
    kernel: str = "rbf",
    pen: float = 1.0,
    max_shift: int = 20,
    min_seg_len: int = 1,
    keep_edge_boundaries_fixed: bool = True,
    min_side_len: int = 1,
    min_improvement: float = 0.0,
) -> tuple[dict[str, Any], dict[str, int | float]]:
    if "signal" not in record or "base_sample_spans_rel" not in record:
        raise ValueError("record 缺少 'signal' 或 'base_sample_spans_rel' 字段")

    signal = validate_signal(record["signal"])
    signal_len = len(signal)

    spans = validate_spans(record["base_sample_spans_rel"], signal_len)
    model_bkps = internal_boundaries_from_spans(spans, signal_len)

    physical_bkps = detect_physical_boundaries(
        signal=signal,
        kernel=kernel,
        pen=pen,
    )

    if mode in {"right", "left"}:
        corrected_bkps, detail_stats = correct_boundaries_directional(
            model_bkps=model_bkps,
            physical_bkps=physical_bkps,
            signal_len=signal_len,
            direction=mode,
            max_shift=max_shift,
            min_seg_len=min_seg_len,
            keep_edge_boundaries_fixed=keep_edge_boundaries_fixed,
        )
    elif mode == "both-best":
        corrected_bkps, detail_stats = correct_boundaries_bidirectional_best(
            model_bkps=model_bkps,
            physical_bkps=physical_bkps,
            signal=signal,
            signal_len=signal_len,
            max_shift=max_shift,
            min_seg_len=min_seg_len,
            keep_edge_boundaries_fixed=keep_edge_boundaries_fixed,
            min_side_len=min_side_len,
            min_improvement=min_improvement,
        )
    else:
        raise ValueError(f"未知 mode: {mode}")

    adj_spans = internal_boundaries_to_spans(corrected_bkps, signal_len)
    adj_spans = validate_spans(adj_spans, signal_len)

    if len(adj_spans) != len(spans):
        raise ValueError(
            f"校正前后 spans 数量不一致: before={len(spans)}, after={len(adj_spans)}"
        )

    new_record = dict(record)
    new_record["base_sample_spans_rel_adj"] = adj_spans

    stats = {
        "signal_len": signal_len,
        "n_spans": len(spans),
        "n_model_bkps": detail_stats["n_model_bkps"],
        "n_physical_bkps": detail_stats["n_physical_bkps"],
        "n_corrected_bkps": len(corrected_bkps),
        "n_mode_changed": detail_stats["n_mode_changed"],
        "n_directional_changed": detail_stats["n_directional_changed"],
        "n_bidirectional_changed": detail_stats["n_bidirectional_changed"],
        "n_enforced_changed": detail_stats["n_enforced_changed"],
        "n_total_changed": detail_stats["n_total_changed"],
        "n_fixed_edges": detail_stats["n_fixed_edges"],
        "n_no_interval_kept": detail_stats["n_no_interval_kept"],
        "n_no_candidate_kept": detail_stats["n_no_candidate_kept"],
        "n_intervals_evaluated": detail_stats["n_intervals_evaluated"],
        "n_candidates_total": detail_stats["n_candidates_total"],
        "n_left_selected": detail_stats["n_left_selected"],
        "n_right_selected": detail_stats["n_right_selected"],
        "n_original_selected": detail_stats["n_original_selected"],
    }

    return new_record, stats


def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    if args.max_shift < 0:
        raise ValueError(f"--max-shift 必须 >= 0，实际为 {args.max_shift}")

    if args.min_seg_len < 1:
        raise ValueError(f"--min-seg-len 必须 >= 1，实际为 {args.min_seg_len}")

    if args.min_side_len < 1:
        raise ValueError(f"--min-side-len 必须 >= 1，实际为 {args.min_side_len}")

    n_total = 0
    n_success = 0
    n_failed = 0

    total_internal_boundaries = 0
    total_physical_boundaries = 0
    total_mode_changed = 0
    total_directional_changed = 0
    total_bidirectional_changed = 0
    total_enforced_changed = 0
    total_changed = 0
    total_fixed_edges = 0
    total_no_interval_kept = 0
    total_no_candidate_kept = 0
    total_intervals_evaluated = 0
    total_candidates = 0
    total_left_selected = 0
    total_right_selected = 0
    total_original_selected = 0

    with smart_open(args.input, "rt", encoding="utf-8") as fin, \
         smart_open(args.output, "wt", encoding="utf-8") as fout:

        for line_idx, line in enumerate(fin, start=1):
            if args.max_reads is not None and n_total >= args.max_reads:
                break

            line = line.strip()
            if not line:
                continue

            n_total += 1

            try:
                record = json.loads(line)

                new_record, stats = adjust_one_record(
                    record=record,
                    mode=args.mode,
                    kernel=args.kernel,
                    pen=args.pen,
                    max_shift=args.max_shift,
                    min_seg_len=args.min_seg_len,
                    keep_edge_boundaries_fixed=args.keep_edge_boundaries_fixed,
                    min_side_len=args.min_side_len,
                    min_improvement=args.min_improvement,
                )

                fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")

                n_success += 1

                total_internal_boundaries += int(stats["n_model_bkps"])
                total_physical_boundaries += int(stats["n_physical_bkps"])
                total_mode_changed += int(stats["n_mode_changed"])
                total_directional_changed += int(stats["n_directional_changed"])
                total_bidirectional_changed += int(stats["n_bidirectional_changed"])
                total_enforced_changed += int(stats["n_enforced_changed"])
                total_changed += int(stats["n_total_changed"])
                total_fixed_edges += int(stats["n_fixed_edges"])
                total_no_interval_kept += int(stats["n_no_interval_kept"])
                total_no_candidate_kept += int(stats["n_no_candidate_kept"])
                total_intervals_evaluated += int(stats["n_intervals_evaluated"])
                total_candidates += int(stats["n_candidates_total"])
                total_left_selected += int(stats["n_left_selected"])
                total_right_selected += int(stats["n_right_selected"])
                total_original_selected += int(stats["n_original_selected"])

            except Exception as e:
                n_failed += 1

                read_id = None
                record_for_failed = None

                try:
                    record_for_failed = json.loads(line)
                    read_id = record_for_failed.get("read_id", None)
                except Exception:
                    record_for_failed = None

                print(
                    f"[WARN] 第 {line_idx} 行处理失败, "
                    f"read_id={read_id}, error={e}"
                )

                if args.write_failed and record_for_failed is not None:
                    fout.write(json.dumps(record_for_failed, ensure_ascii=False) + "\n")

            if args.report_every > 0 and n_total % args.report_every == 0:
                mode_ratio = safe_ratio(total_mode_changed, total_internal_boundaries)
                directional_ratio = safe_ratio(
                    total_directional_changed,
                    total_internal_boundaries,
                )
                bidirectional_ratio = safe_ratio(
                    total_bidirectional_changed,
                    total_internal_boundaries,
                )
                enforced_ratio = safe_ratio(
                    total_enforced_changed,
                    total_internal_boundaries,
                )
                total_ratio = safe_ratio(total_changed, total_internal_boundaries)
                avg_candidates = safe_ratio(total_candidates, total_intervals_evaluated)

                print(
                    f"[INFO] processed={n_total}, success={n_success}, failed={n_failed}, "
                    f"mode={args.mode}, "
                    f"mode_changed_ratio={mode_ratio:.4f}, "
                    f"directional_ratio={directional_ratio:.4f}, "
                    f"bidirectional_ratio={bidirectional_ratio:.4f}, "
                    f"enforce_ratio={enforced_ratio:.4f}, "
                    f"total_changed_ratio={total_ratio:.4f}, "
                    f"left_selected={total_left_selected}, "
                    f"right_selected={total_right_selected}, "
                    f"original_selected={total_original_selected}, "
                    f"fixed_edges={total_fixed_edges}, "
                    f"no_interval_kept={total_no_interval_kept}, "
                    f"no_candidate_kept={total_no_candidate_kept}, "
                    f"avg_candidates_per_interval={avg_candidates:.2f}"
                )

    mode_ratio = safe_ratio(total_mode_changed, total_internal_boundaries)
    directional_ratio = safe_ratio(total_directional_changed, total_internal_boundaries)
    bidirectional_ratio = safe_ratio(total_bidirectional_changed, total_internal_boundaries)
    enforced_ratio = safe_ratio(total_enforced_changed, total_internal_boundaries)
    total_ratio = safe_ratio(total_changed, total_internal_boundaries)
    avg_candidates = safe_ratio(total_candidates, total_intervals_evaluated)
    avg_physical = safe_ratio(total_physical_boundaries, n_success)

    left_select_ratio = safe_ratio(total_left_selected, total_internal_boundaries)
    right_select_ratio = safe_ratio(total_right_selected, total_internal_boundaries)
    original_select_ratio = safe_ratio(total_original_selected, total_internal_boundaries)

    print("=" * 80)
    print(f"[DONE] 输入文件: {args.input}")
    print(f"[DONE] 输出文件: {args.output}")
    print(f"[DONE] mode: {args.mode}")
    print(f"[DONE] kernel: {args.kernel}")
    print(f"[DONE] pen: {args.pen}")
    print(f"[DONE] max_shift: {args.max_shift}")
    print(f"[DONE] min_seg_len: {args.min_seg_len}")
    print(f"[DONE] min_side_len: {args.min_side_len}")
    print(f"[DONE] min_improvement: {args.min_improvement}")
    print(f"[DONE] keep_edge_boundaries_fixed: {args.keep_edge_boundaries_fixed}")
    print("-" * 80)
    print(f"[DONE] 总处理条数: {n_total}")
    print(f"[DONE] 成功条数: {n_success}")
    print(f"[DONE] 失败条数: {n_failed}")
    print(f"[DONE] 总内部边界数: {total_internal_boundaries}")
    print(f"[DONE] 平均每条 read 的 physical 边界数: {avg_physical:.6f}")
    print("-" * 80)
    print(f"[DONE] mode 直接替换修改边界数: {total_mode_changed}")
    print(f"[DONE] 单方向候选替换修改边界数: {total_directional_changed}")
    print(f"[DONE] 双方向 best 替换修改边界数: {total_bidirectional_changed}")
    print(f"[DONE] enforce 修改边界数: {total_enforced_changed}")
    print(f"[DONE] 最终修改边界数: {total_changed}")
    print("-" * 80)
    print(f"[DONE] 固定首尾边界数: {total_fixed_edges}")
    print(f"[DONE] 无合法搜索区间而保留原值数: {total_no_interval_kept}")
    print(f"[DONE] 无候选而保留原值数: {total_no_candidate_kept}")
    print(f"[DONE] 实际评估的局部区间数: {total_intervals_evaluated}")
    print(f"[DONE] 候选变点总数: {total_candidates}")
    print(f"[DONE] 平均每个局部区间候选数: {avg_candidates:.6f}")
    print("-" * 80)
    print(f"[DONE] 选择 left 的边界数: {total_left_selected}")
    print(f"[DONE] 选择 right 的边界数: {total_right_selected}")
    print(f"[DONE] 保留 original 的边界数: {total_original_selected}")
    print(f"[DONE] left 选择比例: {left_select_ratio:.6f}")
    print(f"[DONE] right 选择比例: {right_select_ratio:.6f}")
    print(f"[DONE] original 保留比例: {original_select_ratio:.6f}")
    print("-" * 80)
    print(f"[DONE] mode 直接替换比例: {mode_ratio:.6f}")
    print(f"[DONE] 单方向候选替换比例: {directional_ratio:.6f}")
    print(f"[DONE] 双方向 best 替换比例: {bidirectional_ratio:.6f}")
    print(f"[DONE] enforce 修改比例: {enforced_ratio:.6f}")
    print(f"[DONE] 最终修改比例: {total_ratio:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()