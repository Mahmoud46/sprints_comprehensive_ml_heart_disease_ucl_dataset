from config.paths import FEATURE_SELECTION_DIR

import matplotlib

matplotlib.use("Agg")  # Save plots quietly without pop-ups
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

sns.set_theme(style="whitegrid")


def feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_prefix: str = "feature_selection",
    top_k: int = 8,
):
    """Executes RF Feature Importance, RFE, and Chi-Square tests.

    Saves visualizations and returns the reduced dataset containing the top_k
    consensus features.
    """
    feature_names = X_train.columns.tolist()

    # -------------------------------------------------------------
    # RANDOM FOREST FEATURE IMPORTANCE
    # -------------------------------------------------------------

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    rf_importance = pd.Series(rf.feature_importances_, index=feature_names)
    rf_ranked = rf_importance.sort_values(ascending=False)

    # -------------------------------------------------------------
    # RECURSIVE FEATURE ELIMINATION (RFE)
    # -------------------------------------------------------------
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    rfe = RFE(estimator=log_reg, n_features_to_select=top_k)
    rfe.fit(X_train, y_train)

    rfe_ranking = pd.Series(rfe.ranking_, index=feature_names).sort_values()

    # -------------------------------------------------------------
    # CHI-SQUARE TEST FOR FEATURE SIGNIFICANCE
    # -------------------------------------------------------------
    # Chi-square requires non-negative values, so scale features to [0, 1]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    chi2_selector = SelectKBest(score_func=chi2, k="all")
    chi2_selector.fit(X_train_scaled, y_train)

    chi2_scores = pd.Series(
        chi2_selector.scores_, index=feature_names
    ).sort_values(ascending=False)

    # -------------------------------------------------------------
    # VISUALIZATION: FEATURE IMPORTANCE RANKINGS
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Random Forest Importance
    sns.barplot(
        x=rf_ranked.values,
        y=rf_ranked.index,
        ax=axes[0],
        palette="viridis",
        hue=rf_ranked.index,
        legend=False,
    )
    axes[0].set_title(
        "Random Forest Feature Importance", fontsize=12, fontweight="bold"
    )
    axes[0].set_xlabel("Importance Score")

    # Plot 2: Chi-Square Scores
    sns.barplot(
        x=chi2_scores.values,
        y=chi2_scores.index,
        ax=axes[1],
        palette="magma",
        hue=chi2_scores.index,
        legend=False,
    )
    axes[1].set_title(
        "Chi-Square Test Significance (Chi2 Score)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("Chi2 Statistic Score")

    # Plot 3: RFE Selection (1 = Selected, >1 = Rank)
    rfe_plot_data = rfe_ranking.sort_values()
    sns.barplot(
        x=rfe_plot_data.values,
        y=rfe_plot_data.index,
        ax=axes[2],
        palette="mako",
        hue=rfe_plot_data.index,
        legend=False,
    )
    axes[2].set_title(
        "RFE Ranking (1 = Selected Top Features)", fontsize=12, fontweight="bold"
    )
    axes[2].set_xlabel("Rank (Lower is Better)")

    plt.tight_layout()
    plt.savefig(
        f"{FEATURE_SELECTION_DIR}/{output_prefix}_rankings.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # -------------------------------------------------------------
    # 5. SELECT TOP K CONSENSUS FEATURES & CREATE REDUCED DATASET
    # -------------------------------------------------------------
    # Aggregate normalized ranks across all three methods
    rank_df = pd.DataFrame(
        {
            "RF_Rank": rf_importance.rank(ascending=False),
            "RFE_Rank": rfe.ranking_,
            "Chi2_Rank": chi2_scores.rank(ascending=False),
        }
    )

    rank_df["Consensus_Score"] = rank_df.mean(axis=1)
    consensus_features = rank_df.sort_values("Consensus_Score").index[
        :top_k
    ].tolist()

    print(f"Top {top_k} Consensus Selected Features:")
    for idx, feat in enumerate(consensus_features, 1):
        print(f"  {idx}. {feat}")

    # Return reduced DataFrame
    X_train_reduced = X_train[consensus_features]

    return X_train_reduced, consensus_features