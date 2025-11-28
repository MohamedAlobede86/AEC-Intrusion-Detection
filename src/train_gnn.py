import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load graph data
try:
    graph_data = torch.load("graph_data_safe.pt", weights_only=False)
except FileNotFoundError:
    print("⚠️ Error: graph_data_safe.pt not found. Place it in the repository root.")
    raise SystemExit

NUM_CLASSES = len(torch.unique(graph_data.y))
IN_FEATURES = graph_data.num_features

# 2. Define models (dropout 0.2)
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index); x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index); x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index)

# 3. Create a train/test split over node indices
node_indices = np.arange(graph_data.num_nodes)
_, test_idx, _, _ = train_test_split(
    node_indices, graph_data.y.cpu().numpy(),
    test_size=0.2, stratify=graph_data.y.cpu().numpy(), random_state=42
)
test_mask = torch.tensor(test_idx)

# 4. Load checkpoints if available
def load_model(model, name):
    ckpt = f"{name}_checkpoint.pth"
    if os.path.exists(ckpt):
        state = torch.load(ckpt, weights_only=False)
        model.load_state_dict(state["model_state"])
        model.eval()
        print(f"✅ Loaded {name} checkpoint.")
        return model
    print(f"⚠️ {name} checkpoint not found. Skipping.")
    return None

gcn = load_model(GCN(IN_FEATURES, 64, NUM_CLASSES), "GCN")
sage = load_model(GraphSAGE(IN_FEATURES, 64, NUM_CLASSES), "GraphSAGE")
gat = load_model(GAT(IN_FEATURES, 16, NUM_CLASSES), "GAT")

# 5. Evaluate available models
def evaluate_model(model, name):
    if model is None:
        return None
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        test_out = out[test_mask]
        y_true = graph_data.y[test_mask].cpu().numpy()
        probs = F.softmax(test_out, dim=1).cpu().numpy()
        preds = test_out.argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    auc_macro = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (Macro): {f1:.4f}")
    print(f"AUC (Macro): {auc_macro:.4f}")

    # Confusion matrix (saved to figures/)
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix (Test)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{name}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    return {"name": name, "accuracy": acc, "f1_macro": f1, "auc_macro": auc_macro}

results = []
for m, n in [(gcn,"GCN"), (sage,"GraphSAGE"), (gat,"GAT")]:
    r = evaluate_model(m, n)
    if r: results.append(r)

print("\n✅ GNN evaluation completed and figures saved to ./figures/")
