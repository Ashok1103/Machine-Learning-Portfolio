import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, roc_auc_score,
                             ConfusionMatrixDisplay)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
 
 
def print_results(model_name, y_test, preds, probs):
    print(f'{model_name}')
    print(classification_report(y_test, preds, target_names=['Dropped', 'Enrolled']))
    print(f'AUC-ROC: {roc_auc_score(y_test, probs):.4f}')
 
 
def plot_confusion_matrix(y_test, preds, title, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds,
        display_labels=['Dropped', 'Enrolled'],
        colorbar=False, ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
 
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
 
 
def plot_feature_importance(rf_model, X_train, top_n=20, save_path=None):
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
 
    top = importance_df.head(top_n)
 
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top['Feature'][::-1], top['Importance'][::-1],
            color='steelblue', edgecolor='white')
    ax.set_xlabel('Feature Importance Score')
    ax.set_title(f'Random Forest - Top {top_n} Feature Importances\n(WatSPEED Enrollment Prediction)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
 
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
 
    return importance_df
 
 
def plot_kmeans_elbow(df_model, k_range=range(2, 11), save_path=None):
    import warnings
    warnings.filterwarnings('ignore')
    cluster_cols = [col for col in df_model.columns if
                    col.startswith('Province_') or
                    col.startswith('Program_') or
                    col.startswith('Job Family_') or
                    col == 'Is_Canada']
 
    X_cluster = df_model[cluster_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
 
    inertia_scores = []
    warnings.filterwarnings('ignore')
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia_scores.append(km.inertia_)
 
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_range, inertia_scores, marker='o', color='steelblue', linewidth=2)
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Inertia (Within-Cluster Distance)')
    ax.set_title('Elbow Method — Optimal Number of Clusters')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
 
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
 
    return list(zip(k_range, inertia_scores))