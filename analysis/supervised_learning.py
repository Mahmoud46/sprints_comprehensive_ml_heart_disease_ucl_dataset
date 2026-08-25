
from libs.evaluate_and_compare_models import evaluate_and_compare_models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

class SupervisedLearning:
    def __init__(self, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, pca_optimal, X_train_pca, X_train_reduced, consensus_features):
        self.X_train= X_train 
        self.X_test= X_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.pca_optimal = pca_optimal
        self.X_train_pca = X_train_pca
        self.X_test_pca = self.pca_optimal.fit_transform(self.X_test_scaled)
        self.consensus_features = consensus_features
        self.X_train_reduced = X_train_reduced
        self.X_test_reduced = self.X_test[self.consensus_features]

    def logistic_regression(self, X_train, y_train):
        logistic_regression_model = LogisticRegression()
        logistic_regression_model.fit(X_train, y_train)

        return logistic_regression_model
    
    def decision_tree(self, X_train, y_train):
        decision_tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
        decision_tree_model.fit(X_train, y_train)

        return decision_tree_model

    def random_forest(self, X_train, y_train):
        random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
        random_forest_model.fit(X_train, y_train)

        return random_forest_model
    
    def support_vector_machine(self, X_train, y_train):
        svm_model = SVC(kernel='rbf', random_state=42)
        svm_model.fit(X_train, y_train)

        return svm_model

    def train_test_logistic_regression(self):
        # Training model with pca and features selection
        lr_pca_model = self.logistic_regression(self.X_train_pca, self.y_train)
        lr_fs_model = self.logistic_regression(self.X_train_reduced, self.y_train)

        # Testing model with pca and features selection
        y_pred_lr_pca = lr_pca_model.predict(self.X_test_pca)
        y_pred_lr_fs = lr_fs_model.predict(self.X_test_reduced)

        return lr_pca_model, lr_fs_model, y_pred_lr_pca, y_pred_lr_fs

    def train_test_decision_tree(self):
        # Training model with features selection
        dt_fs_model = self.decision_tree(self.X_train_reduced, self.y_train)

        # Testing model with features selection
        y_pred_dt_fs = dt_fs_model.predict(self.X_test_reduced)

        return dt_fs_model, y_pred_dt_fs

    def train_test_random_forest(self):
        # Training model with features selection
        rf_fs_model = self.random_forest(self.X_train_reduced, self.y_train)

        # Testing model with features selection
        y_pred_rf_fs = rf_fs_model.predict(self.X_test_reduced)

        return rf_fs_model, y_pred_rf_fs

    def train_test_svm(self):
        # Training model with pca
        svm_pca_model = self.support_vector_machine(self.X_train_pca, self.y_train)

        # Testing model with pca
        y_pred_svm_pca = svm_pca_model.predict(self.X_test_pca)

        return svm_pca_model, y_pred_svm_pca

    def supervised_learning(self):
        lr_pca_model, lr_fs_model, y_pred_lr_pca, y_pred_lr_fs = self.train_test_logistic_regression()
        dt_fs_model, y_pred_dt_fs = self.train_test_decision_tree()
        rf_fs_model, y_pred_rf_fs = self.train_test_random_forest()
        svm_pca_model, y_pred_svm_pca = self.train_test_svm()


        model_evaluation_dict = {
            "Logistic Regression (PCA)": {
                "model": lr_pca_model,
                "y_pred": y_pred_lr_pca,
                "X_test": self.X_test_pca,  
            },
            "Logistic Regression (FS)": {
                "model": lr_fs_model,
                "y_pred": y_pred_lr_fs,
                "X_test": self.X_test_reduced, 
            },
            "Decision Tree (FS)": {
                "model": dt_fs_model,
                "y_pred": y_pred_dt_fs,
                "X_test": self.X_test_reduced,
            },
            "Random Forest (FS)": {
                "model": rf_fs_model,
                "y_pred": y_pred_rf_fs,
                "X_test": self.X_test_reduced,
            },
            "SVM (PCA)": {
                "model": svm_pca_model,
                "y_pred": y_pred_svm_pca,
                "X_test": self.X_test_pca,
            },
        }

        results_df = evaluate_and_compare_models(model_evaluation_dict, self.y_test, output_prefix="./output/supervised_learning/model_evaluation")
        return results_df, lr_pca_model, lr_fs_model, dt_fs_model, rf_fs_model, svm_pca_model


        

