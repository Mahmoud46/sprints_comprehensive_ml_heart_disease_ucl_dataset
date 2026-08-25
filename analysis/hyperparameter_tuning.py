import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def hyperparameter_tuning(X_train_pca, X_train_fs, y_train, X_test_pca, X_test_fs, y_test):
    """Executes hyperparameter tuning dynamically using the provided model_configs dictionary

    handling specific feature subsets (FS vs PCA) per model configuration.
    """

    model_configs = {
    "Logistic Regression (FS)": {
        "estimator": LogisticRegression(
            random_state=42, max_iter=1000, class_weight="balanced"
        ),
        "search_type": "grid",
        "param_grid": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "data_type": "fs",
        "X_train": X_train_fs,
        "X_test": X_test_fs
    },
    "Logistic Regression (PCA)": {
        "estimator": LogisticRegression(
            random_state=42, max_iter=1000, class_weight="balanced"
        ),
        "search_type": "grid",
        "param_grid": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "data_type": "pca",
        "X_train": X_train_pca,
        "X_test": X_test_pca
    },
    "Decision Tree (FS)": {
        "estimator": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "search_type": "grid",
        "param_grid": {
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"],
        },
        "data_type": "fs",
        "X_train": X_train_fs,
        "X_test": X_test_fs
    },
    "Random Forest (FS)": {
        "estimator": RandomForestClassifier(
            random_state=42, class_weight="balanced"
        ),
        "search_type": "random",
        "param_distributions": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        },
        "data_type": "fs",
        "X_train": X_train_fs,
        "X_test": X_test_fs
    },
    "SVM (PCA)": {
        "estimator": SVC(
            probability=True, random_state=42, class_weight="balanced"
        ),
        "search_type": "random",
        "param_distributions": {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1, 1],
            "kernel": ["rbf", "linear", "poly"],
        },
        "data_type": "pca",
        "X_train": X_train_pca,
        "X_test": X_test_pca
    },
}

    comparison_records = []
    tuned_models_dict = {}

    print("================ 1. BASELINE EVALUATION & TUNING ================\n")

    for config_name, config in model_configs.items():
        base_estimator = config["estimator"]
        X_tr = config["X_train"]
        X_te = config["X_test"]

        # --- A. Fit Baseline Model ---
        base_estimator.fit(X_tr, y_train)
        y_pred_base = base_estimator.predict(X_te)
        baseline_f1 = f1_score(y_test, y_pred_base, zero_division=0)

        # --- B. Perform Search Strategy (GridSearchCV vs RandomizedSearchCV) ---
        print(
            f"--> Tuning {config_name} using {config['search_type'].upper()}Search..."
        )

        if config["search_type"] == "grid":
            search = GridSearchCV(
                estimator=base_estimator,
                param_grid=config["param_grid"],
                cv=5,
                scoring="f1",
                n_jobs=-1,
            )
        else:
            search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=config["param_distributions"],
                n_iter=20,
                cv=5,
                scoring="f1",
                random_state=42,
                n_jobs=-1,
            )

        search.fit(X_tr, y_train)
        best_tuned_model = search.best_estimator_
        tuned_models_dict[config_name] = best_tuned_model

        # --- C. Evaluate Tuned Model on Corresponding Test Set ---
        y_pred_tuned = best_tuned_model.predict(X_te)
        y_proba_tuned = (
            best_tuned_model.predict_proba(X_te)[:, 1]
            if hasattr(best_tuned_model, "predict_proba")
            else None
        )

        tuned_f1 = f1_score(y_test, y_pred_tuned, zero_division=0)
        tuned_auc = (
            roc_auc_score(y_test, y_proba_tuned)
            if y_proba_tuned is not None
            else np.nan
        )
        f1_improvement = tuned_f1 - baseline_f1

        comparison_records.append(
            {
                "Model Variant": config_name,
                "Data Type": config["data_type"].upper(),
                "Search Method": (
                    "GridSearchCV"
                    if config["search_type"] == "grid"
                    else "RandomizedSearchCV"
                ),
                "Baseline F1": round(baseline_f1, 4),
                "Tuned Test F1": round(tuned_f1, 4),
                "F1 Gain": round(f1_improvement, 4),
                "Test Accuracy": round(accuracy_score(y_test, y_pred_tuned), 4),
                "Test Precision": round(
                    precision_score(y_test, y_pred_tuned, zero_division=0), 4
                ),
                "Test Recall": round(
                    recall_score(y_test, y_pred_tuned, zero_division=0), 4
                ),
                "Test ROC-AUC": round(tuned_auc, 4),
                "Best Hyperparameters": search.best_params_,
            }
        )

    # -------------------------------------------------------------------
    # STEP 2: Compare Optimized Models against Baselines
    # -------------------------------------------------------------------
    comparison_df = pd.DataFrame(comparison_records)

    print("\n================ 2. PERFORMANCE COMPARISON TABLE ================")
    display_cols = [
        "Model Variant",
        "Data Type",
        "Search Method",
        "Baseline F1",
        "Tuned Test F1",
        "F1 Gain",
        "Test ROC-AUC",
        "Test Accuracy",
    ]
    print(comparison_df[display_cols].to_string(index=False))

    # -------------------------------------------------------------------
    # DELIVERABLE: Best Performing Model across FS & PCA datasets
    # -------------------------------------------------------------------
    best_row = comparison_df.sort_values(
        by="Tuned Test F1", ascending=False
    ).iloc[0]
    best_config_name = best_row["Model Variant"]
    best_model_object = tuned_models_dict[best_config_name]
    best_X_test = model_configs[best_config_name]["X_test"]

    print(
        "\n================ 3. DELIVERABLE: BEST PERFORMING MODEL ================"
    )
    print(f"✔️ Top Performing Model:      {best_config_name}")
    print(f"✔️ Dataset Type:             {best_row['Data Type']}")
    print(f"✔️ Selection Metric (Test F1): {best_row['Tuned Test F1']}")
    print(f"✔️ Test ROC-AUC Score:        {best_row['Test ROC-AUC']}")
    print(f"✔️ Improvement Over Baseline: +{best_row['F1 Gain']} F1 Points")
    print("\n✔️ Optimal Hyperparameters:")
    for param_name, param_value in best_row["Best Hyperparameters"].items():
        print(f"   • {param_name}: {param_value}")

    print(
        f"\n--- Final Classification Report ({best_config_name} on Test Set) ---"
    )
    y_pred_best = best_model_object.predict(best_X_test)
    print(classification_report(y_test, y_pred_best))

    return best_model_object, comparison_df


# Execution call:
# best_model, comparison_summary = run_custom_tuning_pipeline(model_configs, y_train, y_test)