#Mashiat Tabassum Khan
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import KarateClub
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx

# Load the dataset
dataset = KarateClub(transform=NormalizeFeatures())
data = dataset[0]

print("Input Features (data.x):")
#print(data.x)

print("Edge Index (data.edge_index):", data.edge_index)
#print(data.edge_index)

# Define the GCN Model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(34, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = torch.nn.Linear(2, 4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

# Initialize the model, optimizer, and loss function
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Log file initialization
log_file = 'training_log.txt'
with open(log_file, 'w') as f:
    f.write("Epoch, Loss\n")

# Function to log to console and file
def log_to_console_and_file(message):
    print(message)  # Print to console
    with open(log_file, 'a') as f:
        f.write(message + '\n')  # Append to log file

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop with logging to both console and file
epochs = 400
for epoch in range(epochs):
    loss = train()
    if epoch % 20 == 0:
        log_to_console_and_file(f'Epoch {epoch}, Loss: {loss:.4f}')

# Simulated output log for demonstration purposes
epoch_logs = []  

# Evaluate the model
def test():
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        correct = (pred == data.y).sum().item()
        acc = correct / data.num_nodes
    return acc

accuracy = test()

# Log the test accuracy to both console and file
log_to_console_and_file(f'Test Accuracy: {accuracy:.4f}')

# Visualizing the graph function
def visualize_graph(data, labels=None):
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(8, 6))
    node_colors = labels if labels is not None else 'lightblue'
    nx.draw(G, with_labels=True, node_color=node_colors, cmap=plt.cm.rainbow, node_size=500)
    plt.show()

# Visualizing the graph before training
visualize_graph(data)

# Visualizing the graph with predicted labels
model.eval()
with torch.no_grad():
    predicted_labels = model(data).argmax(dim=1).numpy()
visualize_graph(data, labels=predicted_labels)

