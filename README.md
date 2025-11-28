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
## 📦 Dataset

This project uses the **CIC-IDS-2017-CNNs-v.1.0** dataset available on Kaggle:

🔗 [Download from Kaggle](https://www.kaggle.com/code/nolovelost/cic-ids-2017-cnns-v-1-0/input)

After downloading, place the file `data_balanced_optimized.parquet` inside the `data/` folder:

