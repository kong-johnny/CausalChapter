# -*- coding: utf-8 -*-
"""LightweightCausalRetriever"""


class LightweightCausalRetriever:
    def __init__(self, retrieval_system, mode="light"):
        self.retrieval_system = retrieval_system
        self.mode = mode

    def ensure_built(self, segments, video_key):
        if self.mode == "off":
            return
        need_rebuild = (
            getattr(self.retrieval_system, "causal_dag", None) is None
            or getattr(self.retrieval_system, "current_video_key", None) != video_key
        )
        if need_rebuild:
            self.retrieval_system.build_causal_structure(segments, video_key=video_key)

    def retrieve(self, current_segment, all_segments, current_idx, top_k=3):
        if self.mode == "off":
            return []
        return self.retrieval_system.retrieve(
            current_segment=current_segment,
            all_segments=all_segments,
            current_idx=current_idx,
            top_k=top_k,
        )