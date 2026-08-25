from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    auc, 
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns
from config.analysis_settings import DATA_COLUMNS_NAMES
from config.paths import FINAL_MODEL_EVALUATION_GRAPHS_DIR, FINAL_MODEL_EVALUATION_METRICS_PATH, DATA_DIR, FINAL_MODEL_PATH

def final_model_evaluation_roc_curve_cm_graphs(y, y_pred, y_prob):
    # Compute Metrics
    cm = confusion_matrix(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    # Create Subplots (1 Row, 2 Columns)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- SUBPLOT 1: CONFUSION MATRIX (sns.heatmap) ---
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["Healthy (0)", "Disease (1+)"],
        yticklabels=["Healthy (0)", "Disease (1+)"],
        ax=axes[0],
        annot_kws={"size": 13, "weight": "bold"},
    )
    axes[0].set_title("Confusion Matrix", weight="bold", fontsize=14, pad=10)
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)

    # --- SUBPLOT 2: ROC CURVE (plt.plot) ---
    axes[1].plot(
        fpr,
        tpr,
        color="#1f77b4",
        linewidth=2.5,
        label=f"Model ROC (AUC = {roc_auc:.4f})",
    )
    axes[1].plot(
        [0, 1],
        [0, 1],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Random Guess (AUC = 0.50)",
    )

    axes[1].set_title("Receiver Operating Characteristic (ROC) Curve", weight="bold", fontsize=14, pad=10)
    axes[1].set_xlabel("False Positive Rate", fontsize=11)
    axes[1].set_ylabel("True Positive Rate", fontsize=11)
    axes[1].set_xlim([-0.02, 1.02])
    axes[1].set_ylim([-0.02, 1.02])
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{FINAL_MODEL_EVALUATION_GRAPHS_DIR}/final_model_roc_curve_cm_evaluation_graphs.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Plots saved successfully to '{FINAL_MODEL_EVALUATION_GRAPHS_DIR}/best_model_roc_curve_cm_evaluation_graphs.png'")


def evaluate_final_model(model, training_dataset_samples, X_test, y_test, consensus_features):
    y_pred = model.predict(X_test)
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
            y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary")
    rec = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    cm = confusion_matrix(y_test, y_pred)
    class_report = classification_report(
        y_test, y_pred, target_names=["Healthy (0)", "Disease (1+)"]
    )
    roc_auc_str = "N/A"
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
        roc_auc_str = f"{roc_auc:.4f}"

    final_model_evaluation_roc_curve_cm_graphs(y_test, y_pred, y_prob)

    metrics_text = f"""==================================================
    UCI HEART DISEASE MODEL EVALUATION METRICS
    ==================================================
    
    1. MODEL & DATASET INFORMATION
    --------------------------------------------------
    Model File       : {FINAL_MODEL_PATH}
    Dataset Evaluated: {DATA_DIR}/processed.cleveland.data
    Total Dateset Samples: {len(X_test) + training_dataset_samples}
    Total Training Samples: {training_dataset_samples}
    Total Testing Samples: {len(X_test)}
    Total Features ({len(DATA_COLUMNS_NAMES) - 1}): {DATA_COLUMNS_NAMES[:-1]}
    Feature Selected ({len(consensus_features)}): {consensus_features}
    
    2. OVERALL PERFORMANCE METRICS
    --------------------------------------------------
    Accuracy         : {acc:.4f} ({acc * 100:.2f}%)
    Precision        : {prec:.4f}
    Recall (Sensitivity): {rec:.4f}
    F1-Score         : {f1:.4f}
    ROC-AUC Score    : {roc_auc_str}
    
    3. CONFUSION MATRIX
    --------------------------------------------------
    [[ True Negatives (TN)   False Positives (FP) ]
     [ False Negatives (FN)  True Positives (TP)  ]]
    
    {cm}
    
    Detailed Counts:
    - True Negatives  (TN) : {cm[0][0]}
    - False Positives (FP) : {cm[0][1]}
    - False Negatives (FN) : {cm[1][0]}
    - True Positives  (TP) : {cm[1][1]}
    
    4. CLASSIFICATION REPORT
    --------------------------------------------------
    {class_report}
    ==================================================
    """
    
    # 7. Print to console and write to txt file
    print("\n" + metrics_text)

    with open(FINAL_MODEL_EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(metrics_text)

    print(f"✅ Evaluation metrics successfully saved to '{FINAL_MODEL_EVALUATION_METRICS_PATH}'")
