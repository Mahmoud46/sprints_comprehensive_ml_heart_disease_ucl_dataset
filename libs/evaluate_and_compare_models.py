import matplotlib

matplotlib.use("Agg")  # Save plots quietly without popups
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

sns.set_theme(style="whitegrid")


def evaluate_and_compare_models(
    model_dict, y_test, output_prefix="model_evaluation"
):
    """Evaluates trained models and saved prediction outputs.

    Generates SEPARATE ROC subplots per model to avoid line overlap, side-by-side
    Confusion Matrices, and prints performance tables.
    """
    results = []
    num_models = len(model_dict)

    # Grid layout parameters (3 columns max)
    cols = 3 if num_models >= 5 else 2
    rows = (num_models + cols - 1) // cols

    # Initialize figures
    fig_roc, axes_roc = plt.subplots(
        rows, cols, figsize=(5.5 * cols, 4.5 * rows), sharex=True, sharey=True
    )
    axes_roc = np.array(axes_roc).flatten()

    fig_cm, axes_cm = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes_cm = np.array(axes_cm).flatten()

    for idx, (name, config) in enumerate(model_dict.items()):
        model = config["model"]
        y_pred = config["y_pred"]
        X_test = config["X_test"]

        # Calculate standard metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Get prediction probabilities/scores for ROC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Calculate Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": roc_auc,
                "TN": cm[0, 0],
                "FP": cm[0, 1],
                "FN": cm[1, 0],
                "TP": cm[1, 1],
            }
        )

        # -------------------------------------------------------------
        # PLOT INDIVIDUAL SEPARATED ROC CURVE
        # -------------------------------------------------------------
        ax_r = axes_roc[idx]
        ax_r.plot(
            fpr,
            tpr,
            color="#1f77b4",
            lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.3f})",
        )
        ax_r.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", lw=1.5)
        ax_r.set_title(name, fontweight="bold", fontsize=11)
        ax_r.set_xlim([0.0, 1.0])
        ax_r.set_ylim([0.0, 1.05])
        ax_r.legend(loc="lower right", fontsize=10)

        # Add axis labels to edge subplots
        if idx >= (rows - 1) * cols:
            ax_r.set_xlabel(
                "False Positive Rate (1 - Specificity)", fontweight="bold"
            )
        if idx % cols == 0:
            ax_r.set_ylabel("True Positive Rate (Recall)", fontweight="bold")

        # -------------------------------------------------------------
        # PLOT CONFUSION MATRIX
        # -------------------------------------------------------------
        ax_c = axes_cm[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax_c,
            cbar=False,
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax_c.set_title(name, fontweight="bold", fontsize=11)
        ax_c.set_xlabel("Predicted Label")
        ax_c.set_ylabel("True Label")
        ax_c.set_xticklabels(["Healthy (0)", "Disease (1)"])
        ax_c.set_yticklabels(["Healthy (0)", "Disease (1)"])

    # Hide unused grid slots
    for i in range(num_models, len(axes_roc)):
        fig_roc.delaxes(axes_roc[i])
        fig_cm.delaxes(axes_cm[i])

    # -------------------------------------------------------------
    # SAVE SEPARATED PLOTS
    # -------------------------------------------------------------
    fig_roc.suptitle(
        "Individual Model ROC Curves Comparison",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig_roc.tight_layout()
    fig_roc.savefig(
        f"{output_prefix}_separated_roc_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_roc)

    fig_cm.suptitle(
        "Confusion Matrix Comparison", fontsize=15, fontweight="bold", y=1.02
    )
    fig_cm.tight_layout()
    fig_cm.savefig(
        f"{output_prefix}_confusion_matrices.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_cm)

    # -------------------------------------------------------------
    # PRINT SUMMARY TABLES
    # -------------------------------------------------------------
    results_df = pd.DataFrame(results).sort_values(
        by="F1-Score", ascending=False
    )
    print("\n=================== MODEL PERFORMANCE SUMMARY ===================")
    print(
        results_df[
            ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        ].to_string(index=False)
    )

    print(
        "\n===================== CONFUSION MATRIX VALUES ====================="
    )
    print(
        results_df[["Model", "TN", "FP", "FN", "TP"]].to_string(index=False)
    )
    print(
        "===================================================================\n"
    )

    return results_df