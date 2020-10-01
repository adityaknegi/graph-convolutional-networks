import torch
from torch.nn.modules.module import Module
from torch.nn import Linear


class GCN(Module):
    """ graph convolutional network"""

    def __init__(self, in_features: int,
                 out_features: int) -> None:
        super(GCN, self).__init__()
        self.lin = Linear(in_features, out_features)

    def forward(self, adj_matrix: torch.Tensor,
                nodes_features: torch.Tensor,
                deg: torch.Tensor) -> torch.Tensor:
        # message passing
        # normalization
        nodes_features = deg.mm(adj_matrix.mm(deg.mm(nodes_features)))
        # linear traonfomation AX^T + B
        nodes_features = self.lin(nodes_features)

        return nodes_features

