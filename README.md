# AEC-Intrusion-Detection
 HEAD

## Project overview
Graph Neural Networks (GCN, GraphSAGE, GAT) and traditional ML (RandomForest, LightGBM) for intrusion detection, with AEC structural explanations for GNNs and SHAP for baselines. Includes high-quality architecture diagrams (landscape and portrait).

## How to run
1. Install dependencies:
   pip install -r requirements.txt

2. Prepare data:
   Place data_balanced_optimized.parquet in ./data/

3. Train/evaluate GNNs:
   python src/train_gnn.py

4. Train/evaluate baselines + SHAP:
   python src/train_baselines.py

5. Generate AEC explanations for a node:
   python src/aec_explain.py

6. Generate architecture diagrams:
   python src/plot_architecture.py
=======
Intrusion-Detection
 7fe951d564c1eee2be4d53a0fa119cdfddfdc2a1
