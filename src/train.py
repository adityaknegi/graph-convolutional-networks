from utiles import *
from model import *
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse

from utiles import get_edges, get_node_info, get_graph, label_encode, accuracy


def run(model) -> typing.Tuple:
    """Method train and validate model"""

    model.train()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.01,
                                 weight_decay=5e-4)

    train_lost = []
    val_lost = []
    train_acc = []
    val_acc = []

    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(adj_matrix, nodes_features, deg)
        loss_train = F.nll_loss(pred[train_index], labels[train_index])
        acc_train = accuracy(pred[train_index], labels[train_index])
        loss_train.backward()
        optimizer.step()

        model.eval()
        loss_val = F.nll_loss(pred[val_index], labels[val_index])
        acc_val = accuracy(pred[val_index], labels[val_index])

        if epoch % 10 == 0:
            print("epoch", epoch, end=" ")
            print("train loss", loss_train.item(), end=" ")
            print("val loss", loss_val.item(), end=" ")

            print("train accu", acc_train.item(), end=" ")
            print("val accu", acc_val.item(), end=" ")
            print()

        train_acc.append(acc_train.item())
        val_acc.append(acc_val.item())
        train_lost.append(loss_train.item())
        val_lost.append(loss_val.item())

    return val_acc, val_lost, train_acc, train_lost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='../cora/', help='Add Dataset', type=str)
    parser.add_argument('--seed', default=102, help='Add Seed', type=int)

    opt = parser.parse_args()
    print(opt)

    file_cities = os.path.join(opt.dataset, 'cora.cites')
    file_content = os.path.join(opt.dataset, 'cora.content')

    edges = get_edges(file_cities)
    nodes_index, nodes_features, labels = get_node_info(file_content)
    G = get_graph(edges=edges)
    label_class, labels = label_encode(labels)

    # get matrix and nodes
    adj_matrix, nodes = nx.attr_matrix(G)
    adj_matrix, nodes = np.asarray(adj_matrix), np.asarray(nodes, dtype=int)

    # degree for each node
    degree = dict(G.degree())

    # add self loop
    np.fill_diagonal(adj_matrix, 1)

    # nodes features in order with adj_matrix nodes
    nodes_features = np.array([nodes_features[nodes_index[str(node)]] for node in nodes])

    # create diagonal matrix
    degree_matrix = np.diag([degree[str(node)] for node in nodes])

    # inverter matrix
    deg = np.linalg.inv(degree_matrix)

    # order liable by nodes
    labels = [labels[nodes_index[str(node)]] for node in nodes]

    device = torch.device('cpu')

    # if gpu available replace device
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = Model(len(nodes_features[0]), len(label_class))

    # add to gpu if device is cuda
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float).to(device)
    nodes_features = torch.tensor(nodes_features, dtype=torch.float).to(device)
    deg = torch.tensor(deg, dtype=torch.float).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    model.to(device)

    train_index, val_index, _, _ = train_test_split(np.arange(0, len(labels)),
                                                    labels,
                                                    random_state=opt.seed,
                                                    test_size=0.2)

    val_acc, val_lost, train_acc, train_lost = run(model)

    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, 'b')
    plt.plot(val_acc, 'r')
    plt.title("train and val accuracy")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.savefig("../img/acc.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_lost,'b')
    plt.plot(val_lost,'r')
    plt.title("train and val loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig("../img/loss.png")
    plt.show()
