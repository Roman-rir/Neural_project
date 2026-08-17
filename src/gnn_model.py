"""GraphSAGE encoder and graph-level classifier."""
from __future__ import annotations
import torch
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * layers
        self.convs = nn.ModuleList(SAGEConv(dims[i], dims[i + 1]) for i in range(layers))
        self.dropout, self.hidden_dim = nn.Dropout(dropout), hidden_dim

    def forward(self, graph) -> torch.Tensor:
        x = graph.x
        for conv in self.convs: x = self.dropout(torch.relu(conv(x, graph.edge_index)))
        batch = getattr(graph, "batch", None)
        if batch is None: batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(x, batch)

class GNNClassifier(nn.Module):
    def __init__(self, encoder: GraphSAGEEncoder, num_labels: int) -> None:
        super().__init__()
        self.encoder, self.head = encoder, nn.Linear(encoder.hidden_dim, num_labels)

    def forward(self, graph) -> torch.Tensor: return self.head(self.encoder(graph))
