"""Full-gallery retrieval with bounded query blocks and conservative tie handling."""
import numpy as np
import argparse
import json


def retrieval_metrics(text, graph, query_batch_size=256):
    text, graph = np.asarray(text), np.asarray(graph)
    if text.ndim != 2 or text.shape != graph.shape or len(text) < 2 or text.shape[1] < 1 or query_batch_size < 1:
        raise ValueError("Require at least two aligned pairs and a positive query batch size")
    if not np.isfinite(text).all() or not np.isfinite(graph).all():
        raise ValueError("Embeddings must be finite")
    text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    graph = graph / np.maximum(np.linalg.norm(graph, axis=1, keepdims=True), 1e-12)
    report = {"gallery_size": len(text), "effective_k": {str(k): min(k, len(text)) for k in (1, 5, 10)},
              "tie_policy": "pessimistic: tied negatives rank ahead of the paired positive"}
    report["random_ranking_expected_recall"] = {str(k): min(k, len(text)) / len(text) for k in (1, 5, 10)}
    for direction, queries, gallery in (("text_to_graph", text, graph), ("graph_to_text", graph, text)):
        ranks = []
        for start in range(0, len(queries), query_batch_size):
            scores = queries[start:start + query_batch_size] @ gallery.T
            positive = scores[np.arange(len(scores)), np.arange(start, start + len(scores))]
            ranks.extend((scores >= positive[:, None]).sum(1).tolist())
        ranks = np.asarray(ranks)
        report[direction] = {f"recall_at_{k}": float(np.mean(ranks <= min(k, len(text)))) for k in (1, 5, 10)}
        report[direction]["median_rank"] = float(np.median(ranks))
        report[direction]["mean_reciprocal_rank"] = float(np.mean(1.0 / ranks))
    report["mean_recall_at_1"] = (report["text_to_graph"]["recall_at_1"] + report["graph_to_text"]["recall_at_1"]) / 2
    return report


def recompute(embeddings, query_batch_size=256):
    """Recompute from saved aligned embeddings without loading a model."""
    with np.load(embeddings, allow_pickle=False) as saved:
        ids = saved["sample_ids"]
        if ids.ndim != 1 or len(ids) != len(saved["text"]) or len(set(ids.tolist())) != len(ids):
            raise ValueError("Saved sample IDs must be unique and aligned with embeddings")
        return retrieval_metrics(saved["text"], saved["graph"], query_batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute retrieval metrics from embeddings.npz")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--query-batch-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(recompute(args.embeddings, args.query_batch_size), indent=2))
