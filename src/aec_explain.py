import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import networkx as nx
import matplotlib.pyplot as plt

# Load graph data
try:
    graph_data = torch.load("graph_data_safe.pt", weights_only=False)
except FileNotFoundError:
    print("⚠️ Error: graph_data_safe.pt not found.")
    raise SystemExit

NUM_CLASSES = len(torch.unique(graph_data.y))
IN_FEATURES = graph_data.num_features

# Minimal GCN for loading checkpoint
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index); x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index)

def load_model_state(model, model_name="GCN"):
    ckpt = f"{model_name}_checkpoint.pth"
    if os.path.exists(ckpt):
        checkpoint = torch.load(ckpt, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        print(f"✅ {model_name} checkpoint loaded for explanation.")
        return model
    print(f"⚠️ {model_name} checkpoint not found.")
    return None

gcn_model = load_model_state(GCN(IN_FEATURES, 64, NUM_CLASSES), "GCN")

def AEC_explain(node_id, model, data, top_k=5):
    if model is None:
        return {"Answer":{"node_id":int(node_id),"prediction":-1,"top_features":[],"neighbors":[]},
                "Evaluate":{"consistency":"Isolated"},
                "Curate":{"note":"Model not loaded"}}
    x = data.x[node_id].detach().cpu().numpy()
    neighbors_mask = (data.edge_index[0] == node_id) | (data.edge_index[1] == node_id)
    connected = data.edge_index.t()[neighbors_mask]
    neighbors = [n.item() for edge in connected for n in edge if n.item() != node_id]
    neighbors = list(set(neighbors))
    with torch.no_grad():
        prediction = model(data.x, data.edge_index)[node_id].argmax().item()
    answer = {
        "node_id": int(node_id),
        "prediction": int(prediction),
        "top_features": x[:top_k].tolist(),
        "neighbors": neighbors[:top_k]
    }
    consistency = "Consistent" if len(neighbors) > 0 else "Isolated"
    return {"Answer": answer, "Evaluate": {"consistency": consistency}, "Curate": {"note": "AEC Structural Explanation"}}

def visualize_explanation(node_id, model, data, top_k=5, save_path=None):
    exp = AEC_explain(node_id, model, data, top_k)
    neighbors_list = exp["Answer"]["neighbors"]
    prediction = exp["Answer"]["prediction"]
    G = nx.Graph()
    G.add_node(node_id, color="red")
    for n in neighbors_list:
        G.add_node(n, color="blue")
        G.add_edge(node_id, n)
    colors = [G.nodes[n]["color"] for n in G.nodes]
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    plt.figure(figsize=(8,6), dpi=300)
    nx.draw(G, pos=pos, with_labels=True, node_color=colors,
            node_size=600, font_size=8,
            labels={node_id: f"{node_id}\nPred: {prediction}"})
    plt.title(f"GCN AEC Explanation for Node {node_id}")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    node_to_explain = 150000
    print(f"--- AEC Explanation for Node {node_to_explain} ---")
    exp = AEC_explain(node_to_explain, gcn_model, graph_data)
    print(json.dumps(exp, indent=2, ensure_ascii=False))
    visualize_explanation(node_to_explain, gcn_model, graph_data, save_path="figures/aec_node_150000.png")
