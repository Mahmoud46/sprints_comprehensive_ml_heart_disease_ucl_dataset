import matplotlib
from kneed import KneeLocator
from config.paths import UNSUPERVISED_LEARNING_DIR

matplotlib.use("Agg")  # Non-interactive backend to save plots without popups
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

sns.set_theme(style="whitegrid")

class UnsupervisedLearning:
    def __init__(self, X_train_scaled, X_test_scaled, y_train, y_test, pca_optimal, X_train_pca):
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.pca_optimal = pca_optimal
        self.X_train_pca = X_train_pca
        self.X_test_pca = self.pca_optimal.fit_transform(self.X_test_scaled)

    def elbow_method(self):
        k_range = range(2, 11)
        inertias = []
        silhouette_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_train_pca)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X_train_pca, kmeans.labels_))

        # Detect optimal_k using KneeLocator
        kl = KneeLocator(
            x=list(k_range), y=inertias, curve="convex", direction="decreasing"
        )
        optimal_k = kl.elbow

        # Plot Elbow Curve + Silhouette Scores with the Knee marked
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Plot 1: Inertia / Elbow ---
        ax1.plot(k_range, inertias, marker="o", color="#1f77b4", lw=2)
        ax1.axvline(
            x=optimal_k,
            color="red",
            linestyle="--",
            lw=1.5,
            label=f"Elbow (K={optimal_k})",
        )
        ax1.set_title("Elbow Method (Inertia vs K)", fontweight="bold")
        ax1.set_xlabel("Number of Clusters (K)")
        ax1.set_ylabel("Inertia (Sum of Squared Distances)")
        ax1.legend(loc="upper right")

        # --- Plot 2: Silhouette Scores ---
        ax2.plot(k_range, silhouette_scores, marker="s", color="#2ca02c", lw=2)
        ax2.axvline(
            x=optimal_k,
            color="red",
            linestyle="--",
            lw=1.5,
            label=f"Selected K ({optimal_k})",
        )
        ax2.set_title("Silhouette Score vs K", fontweight="bold")
        ax2.set_xlabel("Number of Clusters (K)")
        ax2.set_ylabel("Silhouette Score (Higher is Better)")
        ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(
            f"{UNSUPERVISED_LEARNING_DIR}/clustering_kmeans_elbow_silhouette.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


        return optimal_k

    def k_means_clustering(self):
        optimal_k = self.elbow_method()
        kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_clusters = kmeans_optimal.fit_predict(self.X_train_pca)

        return kmeans_optimal, kmeans_clusters, optimal_k

    def dendrogram_analysis(self):
        linkage_matrix = linkage(self.X_train_pca, method="ward")
        hc_labels = fcluster(linkage_matrix, t=8, criterion='distance')
        num_clusters = len(np.unique(hc_labels))

        plt.figure(figsize=(12, 6))
        dendrogram(linkage_matrix, truncate_mode="lastp", p=30, leaf_rotation=90)
        plt.title(
            "Hierarchical Clustering Dendrogram (Ward Linkage)",
            fontsize=13,
            fontweight="bold",
        )
        plt.xlabel("Cluster Size / Sample Index")
        plt.ylabel("Euclidean Distance (Height Cutoff)")
        plt.axhline(
            y=8, color="r", linestyle="--", label=f"Target Split Threshold (K={num_clusters})"
        )
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(
            f"{UNSUPERVISED_LEARNING_DIR}/clustering_hierarchical_dendrogram.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        return num_clusters

    def hierarchical_clustering(self):
        num_clusters = self.dendrogram_analysis()
        hierarchical_optimal = AgglomerativeClustering(n_clusters=num_clusters, linkage="ward")
        hierarchical_clusters = hierarchical_optimal.fit_predict(self.X_train_pca)

        return hierarchical_optimal, num_clusters, hierarchical_clusters

    def visual_comparison(self, kmeans_optimal, kmeans_clusters, num_kmean_clusters, hierarchical_clusters, num_hierarchical_clusters):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)

        # Subplot A: Actual Ground Truth
        sns.scatterplot(
            x=self.X_train_pca[:, 0],
            y=self.X_train_pca[:, 1],
            hue=self.y_train,
            palette="coolwarm",
            ax=axes[0],
            alpha=0.8,
            s=50,
        )
        axes[0].set_title("Ground Truth Labels", fontweight="bold")
        axes[0].set_xlabel("Principal Component 1")
        axes[0].set_ylabel("Principal Component 2")

        # Subplot B: K-Means Derived Clusters
        sns.scatterplot(
            x=self.X_train_pca[:, 0],
            y=self.X_train_pca[:, 1],
            hue=kmeans_clusters,
            palette="Set1",
            ax=axes[1],
            alpha=0.8,
            s=50,
        )
        axes[1].scatter(
            kmeans_optimal.cluster_centers_[:, 0],
            kmeans_optimal.cluster_centers_[:, 1],
            s=200,
            c="yellow",
            marker="X",
            edgecolors="black",
            label="Centroids",
        )
        axes[1].set_title(
            f"K-Means Clustering (K={num_kmean_clusters})", fontweight="bold"
        )
        axes[1].set_xlabel("Principal Component 1")
        axes[1].legend(loc="best")

        # Subplot C: Hierarchical Clustering Derived Clusters
        sns.scatterplot(
            x=self.X_train_pca[:, 0],
            y=self.X_train_pca[:, 1],
            hue=hierarchical_clusters,
            palette="Set2",
            ax=axes[2],
            alpha=0.8,
            s=50,
        )
        axes[2].set_title(
            f"Hierarchical Clustering (K={num_hierarchical_clusters})", fontweight="bold"
        )
        axes[2].set_xlabel("Principal Component 1")

        plt.tight_layout()
        plt.savefig(
            f"{UNSUPERVISED_LEARNING_DIR}/clustering_cluster_vs_ground_truth.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def quantitative_comparison(self, kmeans_clusters, hierarchical_clusters):
        comparison_results = [
            {
                "Algorithm": "K-Means Clustering",
                "Silhouette Score": silhouette_score(self.X_train_pca, kmeans_clusters), # higher is better
                "Adjusted Rand Index (ARI)": adjusted_rand_score(
                    self.y_train, kmeans_clusters
                ),
                "Normalized Mutual Info (NMI)": normalized_mutual_info_score(
                    self.y_train, kmeans_clusters
                ),
            },
            {
                "Algorithm": "Hierarchical Clustering",
                "Silhouette Score": silhouette_score(self.X_train_pca, hierarchical_clusters),
                "Adjusted Rand Index (ARI)": adjusted_rand_score(
                    self.y_train, hierarchical_clusters
                ),
                "Normalized Mutual Info (NMI)": normalized_mutual_info_score(
                    self.y_train, hierarchical_clusters
                ),
            },
        ]

        metrics_df = pd.DataFrame(comparison_results)
        print("\n================ CLUSTERING PERFORMANCE EVALUATION ================")
        print(metrics_df.to_string(index=False))
        print(
            "===================================================================\n"
        )

        return metrics_df

    def unsupervised_learning(self):
        kmeans_optimal, kmeans_clusters, num_kmean_clusters = self.k_means_clustering()
        hierarchical_optimal, num_hierarchical_clusters, hierarchical_clusters = self.hierarchical_clustering()
        self.visual_comparison(kmeans_optimal, kmeans_clusters, num_kmean_clusters, hierarchical_clusters, num_hierarchical_clusters)
        metrics_df = self.quantitative_comparison(kmeans_clusters, hierarchical_clusters)

        return metrics_df, kmeans_clusters, hierarchical_clusters