"""Build deterministic temporal, similarity, and random-control segment graphs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from .features import AudioFeatureConfig, extract_track_features


GRAPH_VARIANTS = ("temporal", "temporal_similarity", "random")


def _validate_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("features must have shape [num_segments, feature_dim]")
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinity")
    return features


def _edge_tensor(edges: set[tuple[int, int]]) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()


def temporal_edge_set(num_nodes: int) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for index in range(num_nodes - 1):
        edges.add((index, index + 1))
        edges.add((index + 1, index))
    return edges


def similarity_edge_set(features: np.ndarray, top_k: int = 2) -> set[tuple[int, int]]:
    """Return symmetric top-k cosine edges between non-adjacent segments."""
    features = _validate_features(features)
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if top_k == 0 or len(features) < 3:
        return set()
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-8)
    similarity = normalized @ normalized.T
    undirected: set[tuple[int, int]] = set()
    for source in range(len(features)):
        candidates = [target for target in range(len(features)) if abs(source - target) > 1]
        candidates.sort(key=lambda target: (-float(similarity[source, target]), target))
        for target in candidates[:top_k]:
            undirected.add((min(source, target), max(source, target)))
    return {(a, b) for edge in undirected for a, b in (edge, edge[::-1])}


def random_edge_set(num_nodes: int, directed_edge_count: int, seed: int) -> set[tuple[int, int]]:
    """Create a symmetric random control with the requested directed edge count."""
    if directed_edge_count < 0 or directed_edge_count % 2:
        raise ValueError("directed_edge_count must be a non-negative even number")
    candidates = [(a, b) for a in range(num_nodes) for b in range(a + 2, num_nodes)]
    needed = min(directed_edge_count // 2, len(candidates))
    chosen = random.Random(seed).sample(candidates, needed) if needed else []
    return {(a, b) for edge in chosen for a, b in (edge, edge[::-1])}


def build_edge_variants(
    features: np.ndarray, *, top_k: int = 2, seed: int = 42
) -> dict[str, torch.Tensor]:
    features = _validate_features(features)
    temporal = temporal_edge_set(len(features))
    similarity = similarity_edge_set(features, top_k)
    random_control = random_edge_set(len(features), len(similarity), seed)
    return {
        "temporal": _edge_tensor(temporal),
        "temporal_similarity": _edge_tensor(temporal | similarity),
        "random": _edge_tensor(temporal | random_control),
    }


def build_segment_graph(
    features: np.ndarray,
    *,
    variant: str = "temporal_similarity",
    top_k: int = 2,
    seed: int = 42,
) -> Data:
    features = _validate_features(features)
    if variant not in GRAPH_VARIANTS:
        raise ValueError(f"variant must be one of {GRAPH_VARIANTS}")
    edge_index = build_edge_variants(features, top_k=top_k, seed=seed)[variant]
    return Data(x=torch.from_numpy(features), edge_index=edge_index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--variant", choices=GRAPH_VARIANTS, default="temporal_similarity")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = AudioFeatureConfig(
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
    )
    features = extract_track_features(args.audio, config)
    graph = build_segment_graph(
        features, variant=args.variant, top_k=args.top_k, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "variant": args.variant,
                "nodes": graph.num_nodes,
                "edges": graph.num_edges,
                "feature_dim": graph.num_node_features,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
