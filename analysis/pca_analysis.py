from config.paths import PCA_DIR
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend to save without showing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid")

def pca_analysis(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    output_prefix: str = "pca",
    variance_threshold: float = 0.95,
):
    # Full PCA to compute cumulative variance
    pca_full = PCA().fit(X_train_scaled)
    cum_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # Determine optimal components for threshold
    n_optimal = np.argmax(cum_variance >= variance_threshold) + 1
    
    print(f"Original Feature Count: {X_train_scaled.shape[1]}")
    print(
        f"Components needed for {variance_threshold:.0%} variance: {n_optimal}"
    )

    # Cumulative explained variance plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(cum_variance) + 1),
        cum_variance,
        marker="o",
        linestyle="--",
        color="b",
        label="Cumulative Variance",
    )
    plt.axhline(
        y=variance_threshold,
        color="r",
        linestyle=":",
        label=f"{variance_threshold:.0%} Variance Cutoff",
    )
    plt.axvline(
        x=n_optimal,
        color="g",
        linestyle="--",
        label=f"Optimal Components ({n_optimal})",
    )

    plt.xlabel("Number of Principal Components", fontweight="bold")
    plt.ylabel("Cumulative Explained Variance Ratio", fontweight="bold")
    plt.title("PCA Explained Variance vs. Number of Components", fontsize=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        f"{PCA_DIR}/{output_prefix}_cumulative_variance.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # -------------------------------------------------------------
    # 2D SCATTER PLOT (First 2 Principal Components)
    # -------------------------------------------------------------
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_train_scaled)

    var_pc1 = pca_2d.explained_variance_ratio_[0] * 100
    var_pc2 = pca_2d.explained_variance_ratio_[1] * 100

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        X_pca_2d[:, 0],
        X_pca_2d[:, 1],
        c=y_train,
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="k",
        linewidths=0.5,
    )
    plt.xlabel(f"Principal Component 1 ({var_pc1:.1f}% Variance)")
    plt.ylabel(f"Principal Component 2 ({var_pc2:.1f}% Variance)")
    plt.title(
        "2D Projection of Heart Disease Dataset via PCA",
        fontsize=12,
        fontweight="bold",
    )

    # Add legend manually for binary targets
    handles, _ = scatter.legend_elements()
    plt.legend(
        handles, ["Healthy (0)", "Disease (1)"], title="Class", loc="best"
    )

    plt.tight_layout()
    plt.savefig(f"{PCA_DIR}/{output_prefix}_scatter_2d.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Apply PCA with optimal number of components
    pca_optimal = PCA(n_components=n_optimal)
    X_train_pca = pca_optimal.fit_transform(X_train_scaled)

    return pca_optimal, X_train_pca