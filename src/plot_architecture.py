import os
import matplotlib.pyplot as plt
import networkx as nx

def plot_architecture(save_path=None, orientation="landscape"):
    G = nx.DiGraph()

    nodes = [
        "Raw Data (Features & Labels)",
        "Train/Test Split",
        "GNN Path: Graph Construction",
        "GNN Path: GCN/SAGE/GAT Training",
        "Traditional Path: RF/XGBoost Training",
        "GNN Evaluation (AUC: 0.98)",
        "GNN Explanation (AEC)",
        "Traditional Explanation (SHAP)",
        "Final Comparison & Reporting"
    ]
    edges = [
        ("Raw Data (Features & Labels)", "Train/Test Split"),
        ("Train/Test Split", "GNN Path: Graph Construction"),
        ("GNN Path: Graph Construction", "GNN Path: GCN/SAGE/GAT Training"),
        ("GNN Path: GCN/SAGE/GAT Training", "GNN Evaluation (AUC: 0.98)"),
        ("GNN Path: GCN/SAGE/GAT Training", "GNN Explanation (AEC)"),
        ("Train/Test Split", "Traditional Path: RF/XGBoost Training"),
        ("Traditional Path: RF/XGBoost Training", "GNN Evaluation (AUC: 0.98)"),
        ("Traditional Path: RF/XGBoost Training", "Traditional Explanation (SHAP)"),
        ("GNN Evaluation (AUC: 0.98)", "Final Comparison & Reporting"),
        ("GNN Explanation (AEC)", "Final Comparison & Reporting"),
        ("Traditional Explanation (SHAP)", "Final Comparison & Reporting")
    ]
    G.add_nodes_from(nodes); G.add_edges_from(edges)

    if orientation == "landscape":
        plt.figure(figsize=(11, 6), dpi=300)
        pos = {
            "Raw Data (Features & Labels)": (0, 3),
            "Train/Test Split": (2, 3),
            "GNN Path: Graph Construction": (4, 4),
            "GNN Path: GCN/SAGE/GAT Training": (6, 4),
            "GNN Evaluation (AUC: 0.98)": (8, 3),
            "GNN Explanation (AEC)": (8, 5),
            "Traditional Path: RF/XGBoost Training": (6, 2),
            "Traditional Explanation (SHAP)": (8, 1),
            "Final Comparison & Reporting": (10, 3)
        }
    else:
        plt.figure(figsize=(7, 11), dpi=300)
        pos = {
            "Raw Data (Features & Labels)": (5, 10),
            "Train/Test Split": (5, 9),
            "GNN Path: Graph Construction": (3, 8),
            "GNN Path: GCN/SAGE/GAT Training": (3, 7),
            "GNN Evaluation (AUC: 0.98)": (5, 6),
            "GNN Explanation (AEC)": (1, 6),
            "Traditional Path: RF/XGBoost Training": (7, 8),
            "Traditional Explanation (SHAP)": (9, 6),
            "Final Comparison & Reporting": (5, 4)
        }

    node_colors = {
        "Raw Data (Features & Labels)": "#FFC0CB",
        "Train/Test Split": "#ADD8E6",
        "GNN Path: Graph Construction": "#87CEFA",
        "GNN Path: GCN/SAGE/GAT Training": "#1E90FF",
        "GNN Evaluation (AUC: 0.98)": "#32CD32",
        "GNN Explanation (AEC)": "#90EE90",
        "Traditional Path: RF/XGBoost Training": "#FFA07A",
        "Traditional Explanation (SHAP)": "#F08080",
        "Final Comparison & Reporting": "#FFD700"
    }
    colors = [node_colors[node] for node in G.nodes]

    nx.draw(G, pos,
            with_labels=True,
            node_color=colors,
            node_size=5000,
            font_size=12,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            width=2,
            connectionstyle='arc3,rad=0.1',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

    plt.title("Architecture Diagram: GNN vs. Traditional ML for Cybersecurity", fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs("figures", exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    plot_architecture(save_path="figures/architecture_diagram_landscape.png", orientation="landscape")
    plot_architecture(save_path="figures/architecture_diagram_portrait.png", orientation="portrait")
    print("✅ Diagrams saved to ./figures/")
