from graphconvnet import *
import torch.nn.functional as F


class Model(Module):

    def __init__(self, num_node_features: int,
                 num_classes: int,
                 hidden_nodes=16) -> None:
        """ num_node_features: input feature size
            num_classes: no of class
            hidden_nodes: hidden unit
        """
        super(Model, self).__init__()

        self.gcn1 = GCN(num_node_features, 16)
        self.gcn2 = GCN(16, num_classes)

    def forward(self,
                adj_matrix,
                nodes_features,
                deg) -> torch.Tensor:
        nodes_features = self.gcn1(adj_matrix, nodes_features, deg)
        nodes_features = F.relu(nodes_features)
        nodes_features = F.dropout(nodes_features, training=self.training)
        nodes_features = self.gcn2(adj_matrix, nodes_features, deg)

        return F.log_softmax(nodes_features, dim=1)