
from config.analysis_settings import NUMERICAL_FEATURES
from config.paths import EDA_DIR
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["font.size"] = 10

def eda(df):
  
    # Histogram (Distribution of Continuous Features)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(NUMERICAL_FEATURES):
        sns.histplot(
            df[col], kde=True, ax=axes[i], color="skyblue", bins=20, edgecolor="black"
        )
        axes[i].set_title(f"Distribution of {col.upper()}", fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    # Plot target balance in 6th subplot
    sns.countplot(x="num", data=df, ax=axes[5], palette="Set2")
    axes[5].set_title("Target Distribution (0 = Healthy, 1 = Disease)", fontweight="bold")
    axes[5].set_xticklabels(["Healthy (0)", "Disease (1)"])

    plt.tight_layout()
    # Save plot
    plt.savefig(f"{EDA_DIR}/eda_histograms.png", dpi=300, bbox_inches="tight")
    plt.close()

    # BOXPLOTS (Outlier Detection & Target Relationship)

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))

    for i, col in enumerate(NUMERICAL_FEATURES):
        sns.boxplot(
            x="num", y=col, data=df, ax=axes[i], palette="Set2", width=0.4
        )
        axes[i].set_title(f"{col.upper()} vs Target", fontweight="bold")
        axes[i].set_xticklabels(["Healthy", "Disease"])

    plt.tight_layout()
    # Save plot 
    plt.savefig(f"{EDA_DIR}/eda_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close()


    # Correlation Heatmap (Linear Relationships)
    plt.figure(figsize=(12, 8))

    # Compute correlation matrix on clean numeric values
    corr_matrix = df.drop(columns=["num"]).corr()

    # Mask upper triangle for clarity
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Cleveland Dataset Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    # Save plot
    plt.savefig(f"{EDA_DIR}/eda_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("All EDA plots saved successfully to disk!")