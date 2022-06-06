from torch_geometric.datasets import NELL
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch

dataset = NELL(root='dataset')

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
data = dataset[0]
data1 = dataset[0]  # Get the first graph object.

print()
print(data1)
print('======================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data1.num_nodes}')
print(f'Number of edges: {data1.num_edges}')
print(f'Average node degree: {data1.num_edges / data1.num_nodes:.2f}')
print(f'Number of training nodes: {data1.train_mask.sum()}')
print(f'Training node label rate: {int(data1.train_mask.sum()) / data1.num_nodes:.2f}')
print(f'Contains isolated nodes: {data1.contains_isolated_nodes()}')
print(f'Contains self-loops: {data1.contains_self_loops()}')
print(f'Is undirected: {data1.is_undirected()}')

import torch
from torch.nn import Linear
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(MLP, self).__init__()
        torch.manual_seed(12345)

        # self.conv1 = GCNConv(dataset.num_features, hidden_channels[0])
        # self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
        self.conv1 = GATConv(dataset.num_features, hidden_channels[0])
        self.conv2 = GATConv(hidden_channels[0], hidden_channels[1])
        self.lin1 = Linear(hidden_channels[1], hidden_channels[2])
        self.lin2 = Linear(hidden_channels[2], dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x.to_dense().to(device), edge_index.to(device))
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index.to(device))
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


model = MLP(hidden_channels=[256, 256, 256]).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) 


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    return loss


def test(val):
    if val:
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu()
        test_correct = pred[data.val_mask] == data.y[data.val_mask]
        test_acc = int(test_correct.sum()) / int(data.val_mask.sum())
    else:
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu()
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


for epoch in range(1, 201):
    loss = train()
    if epoch % 10 == 0:
        torch.save(model.state_dict(), str(epoch) + '.pth')
        test_acc = test(True)
        print(f'Val Accuracy: {test_acc:.4f}')
        test_acc = test(False)
        print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
test_acc = test(False)
print(f'Test Accuracy: {test_acc:.4f}')