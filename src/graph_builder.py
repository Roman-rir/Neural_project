"""Build temporal-plus-similarity PyG graphs from audio segments."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from .audio_features import AudioFeatureConfig, extract_track_features

def build_segment_graph(features: np.ndarray, similarity_threshold: float = 0.85) -> Data:
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("features must have shape [num_segments, feature_dim]")
    normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    similarity = normalized @ normalized.T
    edges: set[tuple[int, int]] = set()
    for i in range(len(features)):
        for j in (i - 1, i + 1):
            if 0 <= j < len(features): edges.add((i, j))
        for j in np.where((similarity[i] >= similarity_threshold) & (np.arange(len(features)) != i))[0]: edges.add((i, int(j)))
    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    return Data(x=torch.from_numpy(features), edge_index=edge_index)

def main() -> None:
    parser = argparse.ArgumentParser(description="Build an audio segment graph.")
    parser.add_argument("--audio", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--segment-seconds", type=float, default=5.0); parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    graph = build_segment_graph(extract_track_features(args.audio, AudioFeatureConfig(segment_seconds=args.segment_seconds)), args.threshold)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); torch.save(graph, output)
    print(f"Saved graph with {graph.num_nodes} nodes and {graph.num_edges} edges to {output}")

if __name__ == "__main__": main()
