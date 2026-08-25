import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


import joblib
from hyperparameter_tuning import hyperparameter_tuning
from data_preprocessing import data_preprocessing
from pca_analysis import pca_analysis
from feature_selection import feature_selection
from supervised_learning import SupervisedLearning
from unsupervised_learning import UnsupervisedLearning
from config.paths import FINAL_MODEL_PATH
from libs.evaluate_model import evaluate_final_model

X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = data_preprocessing()

pca_optimal, X_train_pca = pca_analysis(X_train_scaled, y_train)

X_train_reduced, consensus_features = feature_selection( X_train, y_train)

supervised_learning = SupervisedLearning(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, pca_optimal, X_train_pca, X_train_reduced, consensus_features)
results_df, lr_pca_model, lr_fs_model, dt_fs_model, rf_fs_model, svm_pca_model = supervised_learning.supervised_learning()

unsupervised_learning = UnsupervisedLearning(X_train_scaled, X_test_scaled, y_train, y_test, pca_optimal, X_train_pca)
metrics_df, kmeans_clusters, hierarchical_clusters = unsupervised_learning.unsupervised_learning()

best_model_object, comparison_df = hyperparameter_tuning(X_train_pca, X_train[consensus_features], y_train, pca_optimal.fit_transform(X_test_scaled), X_test[consensus_features], y_test)

# -----------------------------------------
# Evaluate and save best model evaluation report and graphs
evaluate_final_model(best_model_object, len(X_train), X_test[consensus_features], y_test, consensus_features)

# Model Export & Deployment
joblib.dump(best_model_object, FINAL_MODEL_PATH)
print(f"✔️ Model successfully saved to {FINAL_MODEL_PATH}")
