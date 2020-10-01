import typing
import numpy as np
from sklearn import preprocessing
import networkx as nx


def get_edges(file_cities: str) -> np.array:
    """ Method return all edges array"""

    edges = []
    with open(file=file_cities, mode='r') as f:
        for data in f.readlines():
            data = data.split('\t')
            edges.append((data[0], data[1][:-1]))

    return np.asarray(edges)


def get_node_info(file_content: str) -> np.array:
    """Method return node name and index, nodes features and lables"""

    nodes_index = {}
    nodes_features = []
    labels = []

    with open(file=file_content, mode='r') as f:
        for i, data in enumerate(f.readlines()):
            data = data.split('\t')
            nodes_index[data[0]] = i
            nodes_features.append(data[1:-1])
            labels.append(data[-1][:-1])

    return nodes_index, np.asarray(nodes_features, dtype=int), np.asarray(labels)


def label_encode(lables: typing.List) -> typing.Tuple:
    """ Method for lable encode"""

    le = preprocessing.LabelEncoder()
    lables = le.fit_transform(lables)

    return le.classes_, lables


def get_graph(edges: typing.List[typing.Set]):
    """ Method to get Graph"""

    G = nx.Graph()
    # G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    print('Graph info: ', nx.info(G))

    return G


def accuracy(pred, labels):
    """ accuracy calculation """
    preds = pred.max(1)[1].type_as(labels)
    preds = preds.eq(labels).double()
    preds = preds.sum()
    return preds / len(labels)
