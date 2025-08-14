# Mashiat Tabassum Khan
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import logging
import random

# Setup logging to print and save to file
log_file_path = 'gnn_PubMed.log'
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler(log_file_path),
    logging.StreamHandler()
])

# Load the PubMed dataset
dataset = Planetoid(root='data', name='PubMed')
data = dataset[0]

def visualize_graph(data, dataset, num_sampled_nodes=500):
    import matplotlib.patches as mpatches

    sampled_nodes = random.sample(range(data.num_nodes), num_sampled_nodes)
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[sampled_nodes] = True

    sub_data = data.subgraph(mask)

    G = to_networkx(sub_data, to_undirected=True)
    y = sub_data.y.cpu().numpy()
    cmap = plt.get_cmap('tab10')
    colors = [cmap(label) for label in y]

    # Define class names (PubMed has 3 classes)
    class_names = {
        0: "Diabetes Mellitus",
        1: "Oncology",
        2: "Cardiovascular Diseases"
    }

    # Generate legend handles
    legend_handles = [
        mpatches.Patch(color=cmap(cls), label=f"{cls}: {name}")
        for cls, name in class_names.items()
    ]

    # Plot
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=colors, node_size=50, edge_color='gray', with_labels=False)
    plt.title("PubMed Subgraph Visualization (Nodes Colored by Class)")
    plt.legend(handles=legend_handles, loc='upper right', fontsize='small')
    plt.show()

# Print dataset info
num_nodes = data.num_nodes
num_edges = data.num_edges
num_classes = dataset.num_classes
logging.info(f"Dataset: PubMed")
logging.info(f"Number of classes: {num_classes}")
logging.info(f"Number of nodes: {num_nodes}")
logging.info(f"Number of edges: {num_edges}")

# Define the GCN model
class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Instantiate model, loss, optimizer
model = GNNModel(in_channels=dataset.num_node_features,
                 hidden_channels=64,
                 out_channels=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
def evaluate():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        val_correct = (pred[data.val_mask] == data.y[data.val_mask]).sum().item()
        val_acc = val_correct / data.val_mask.sum().item()

        test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
        test_acc = test_correct / data.test_mask.sum().item()

    return val_acc, test_acc

# Print model summary
logging.info(f"\nModel:\n{model}")

# Training loop
logging.info("\nTraining started...")
for epoch in range(1, 501):
    loss = train()
    val_acc, test_acc = evaluate()
    logging.info(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    final_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
    final_test_acc = final_correct / data.test_mask.sum().item()

logging.info(f"\n Final Test Accuracy: {final_test_acc:.4f}")

# Visualize class-colored subgraph
visualize_graph(data, dataset)
