# -*- coding: utf-8 -*-
"""Visual/OCR feature indexing and loading.

The original loader silently returned zeros for the available AVLecture caches
because the cached features are often shaped ``[1, D]`` (not ``[D]``) and the
example ids may be stored with different separators, e.g.
``mit049@@MIT8_...`` in jsonl vs ``mit049--MIT8_...`` in the OCR cache.
This module intentionally accepts all common forms so cross-attn actually sees
non-zero visual information.
"""
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def build_vis_index(vis_cache_dir):
    vis_index = defaultdict(dict)
    vis_cache_dir = Path(vis_cache_dir)
    if not vis_cache_dir.exists():
        print(f"[WARN] vis cache dir not found: {vis_cache_dir}")
        return vis_index

    files = list(vis_cache_dir.glob("*.npy"))
    for f in tqdm(files, desc="Index visual features"):
        stem = f.stem
        if "-" not in stem:
            continue
        video_id, utt_idx_str = stem.rsplit("-", 1)
        try:
            utt_idx = int(utt_idx_str)
        except ValueError:
            continue
        vis_index[video_id][utt_idx] = str(f)
    return vis_index


def _candidate_video_keys(example_id):
    eid = str(example_id)
    # Pred-augmented training rows share the same cached visual stream.
    if eid.endswith("__predaug"):
        eid = eid[: -len("__predaug")]
    keys = [
        eid,
        eid.replace("@@", "_"),
        eid.replace("@@", "--"),
        eid.split("@@")[-1],
    ]
    # Preserve order and remove duplicates.
    out = []
    seen = set()
    for k in keys:
        if k and k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _coerce_feature(arr, vis_dim):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        feat = arr
    elif arr.ndim == 2:
        # Common cache format is [1, D].  If multiple frame features are stored
        # for one utterance, mean-pool them to one utterance feature.
        if arr.shape[0] == 0:
            return None
        feat = arr.mean(axis=0)
    else:
        feat = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    if feat.shape[-1] != vis_dim:
        return None
    return feat.astype(np.float32, copy=False)


def _looks_timestamp_indexed(video_map):
    """Heuristic: OCR cache keys are seconds (mostly 10s-spaced), not utt ids."""
    keys = sorted(int(k) for k in video_map.keys())
    if len(keys) < 4:
        return False
    # Utterance caches are usually dense consecutive ids; OCR/video-time caches
    # are sparse (10,20,...) and often run into the thousands of seconds.
    if max(keys) >= 1000:
        return True
    diffs = np.diff(keys).astype(np.float32)
    median_gap = float(np.median(diffs)) if len(diffs) else 1.0
    multiple_10_ratio = sum(1 for k in keys if k % 10 == 0) / max(1, len(keys))
    dense_ratio = sum(1 for d in diffs if d == 1) / max(1, len(diffs))
    return (median_gap >= 5.0 and multiple_10_ratio >= 0.6 and dense_ratio < 0.5)


def _load_by_utterance(video_map, start_idx, end_idx, vis_dim):
    feats = []
    for i in range(int(start_idx), int(end_idx)):
        p = video_map.get(i, None)
        if p is not None and os.path.exists(p):
            try:
                feat = _coerce_feature(np.load(p), vis_dim)
                if feat is not None:
                    feats.append(feat)
            except Exception:
                pass
    return feats


def _load_by_time(video_map, start_time, end_time, vis_dim):
    feats = []
    if start_time is None or end_time is None:
        return feats
    try:
        st = float(start_time)
        ed = float(end_time)
        if ed < st:
            st, ed = ed, st
        # Include frames in [st, ed], and if very short/missing include
        # the nearest following timestamp to avoid all-zero segments.
        keys = sorted(video_map.keys())
        chosen = [k for k in keys if st <= float(k) <= ed]
        if not chosen:
            chosen = [k for k in keys if float(k) >= st][:1]
        for k in chosen:
            p = video_map.get(k)
            if p is not None and os.path.exists(p):
                feat = _coerce_feature(np.load(p), vis_dim)
                if feat is not None:
                    feats.append(feat)
    except Exception:
        pass
    return feats


def load_vis_seq(
    vis_index,
    example_id,
    start_idx,
    end_idx,
    vis_dim=512,
    start_time=None,
    end_time=None,
):
    """Load a segment-level visual sequence.

    Supports two cache styles:
    - utterance-indexed: ``video-utt_idx.npy`` (dense integer ids)
    - timestamp-indexed OCR: ``video-second.npy`` (sparse 10s-ish ids)

    When timestamps are available and the cache looks timestamp-indexed, prefer
    the time window.  This avoids a subtle bug where OCR files named ``-10`` or
    ``-20`` were accidentally treated as utterance ids for segments spanning
    utterances 0..20.
    """
    feats = []
    video_map = None
    for key in _candidate_video_keys(example_id):
        if key in vis_index:
            video_map = vis_index[key]
            break

    if video_map is not None:
        timestamp_like = _looks_timestamp_indexed(video_map)
        if timestamp_like and start_time is not None and end_time is not None:
            feats = _load_by_time(video_map, start_time, end_time, vis_dim)
            if len(feats) == 0:
                feats = _load_by_utterance(video_map, start_idx, end_idx, vis_dim)
        else:
            feats = _load_by_utterance(video_map, start_idx, end_idx, vis_dim)
            if len(feats) == 0:
                feats = _load_by_time(video_map, start_time, end_time, vis_dim)

    if len(feats) == 0:
        feats = [np.zeros((vis_dim,), dtype=np.float32)]
    return np.stack(feats, axis=0).astype(np.float32, copy=False)
