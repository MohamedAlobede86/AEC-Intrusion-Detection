import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import os

# 1) Load dataset
try:
    data = pd.read_parquet("data/data_balanced_optimized.parquet")
except FileNotFoundError:
    print("⚠️ Error: data/data_balanced_optimized.parquet not found.")
    raise SystemExit

# 2) Features and labels
X = data.drop(columns=["Label"])
y = pd.factorize(data["Label"])[0]

# Clean columns
X.columns = (
    X.columns.str.strip()
             .str.replace(" ", "_")
             .str.replace("/", "_")
             .str.replace("-", "_")
)
X = X.select_dtypes(include=[np.number])

# 3) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Train models
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
print("... Training RandomForest")
rf.fit(X_train, y_train)

lgb_model = lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)
print("... Training LightGBM")
lgb_model.fit(X_train, y_train)

# 5) Results
rf_preds = rf.predict(X_test)
lgb_preds = lgb_model.predict(X_test)
rf_probs = rf.predict_proba(X_test)
lgb_probs = lgb_model.predict_proba(X_test)

baseline = pd.DataFrame([
    {
        "Model": "RandomForest",
        "Accuracy": accuracy_score(y_test, rf_preds),
        "Precision": precision_score(y_test, rf_preds, average="macro", zero_division=0),
        "Recall": recall_score(y_test, rf_preds, average="macro", zero_division=0),
        "F1-Score": f1_score(y_test, rf_preds, average="macro", zero_division=0),
        "AUC (Macro)": roc_auc_score(y_test, rf_probs, multi_class='ovr', average='macro')
    },
    {
        "Model": "LightGBM",
        "Accuracy": accuracy_score(y_test, lgb_preds),
        "Precision": precision_score(y_test, lgb_preds, average="macro", zero_division=0),
        "Recall": recall_score(y_test, lgb_preds, average="macro", zero_division=0),
        "F1-Score": f1_score(y_test, lgb_preds, average="macro", zero_division=0),
        "AUC (Macro)": roc_auc_score(y_test, lgb_probs, multi_class='ovr', average='macro')
    }
])
print("\n📊 Traditional Model Results:")
print(baseline.to_markdown(index=False, floatfmt=".4f"))

# 6) SHAP (sample)
sample_n = min(500, int(0.1 * len(X_test)))
sample_idx = np.random.RandomState(42).choice(len(X_test), size=sample_n, replace=False)
sample_X_test = X_test.iloc[sample_idx]

# LightGBM SHAP
print("\n🔍 SHAP Explanation for LightGBM")
explainer_lgb = shap.TreeExplainer(lgb_model.booster_, feature_perturbation="interventional")
shap_values_lgb = explainer_lgb.shap_values(sample_X_test, check_additivity=False)
shap_values_to_plot_lgb = shap_values_lgb[0] if isinstance(shap_values_lgb, list) and len(shap_values_lgb) > 1 else shap_values_lgb

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values_to_plot_lgb, sample_X_test, plot_type="bar", max_display=10, show=False)
plt.title("LightGBM SHAP Feature Importance (Top 10)")
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/shap_lightgbm_top10.png", dpi=300, bbox_inches="tight")
plt.show()

# RandomForest SHAP
print("\n🔍 SHAP Explanation for RandomForest")
explainer_rf = shap.TreeExplainer(rf, feature_perturbation="interventional")
shap_values_rf = explainer_rf.shap_values(sample_X_test, check_additivity=False)
shap_values_to_plot_rf = shap_values_rf[0] if isinstance(shap_values_rf, list) and len(shap_values_rf) > 1 else shap_values_rf

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values_to_plot_rf, sample_X_test, plot_type="bar", max_display=10, show=False)
plt.title("RandomForest SHAP Feature Importance (Top 10)")
plt.savefig("figures/shap_randomforest_top10.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n✅ Baseline training and SHAP figures saved to ./figures/")
