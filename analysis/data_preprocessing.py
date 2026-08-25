import pandas as pd
from config.analysis_settings import DATA_COLUMNS_NAMES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from config.paths import DATA_CLEVELAND_PATH

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# from libs.utils import eda


def data_preprocessing():
    # Load Data and Handle Missing Values
    cleveland_pd = pd.read_csv(DATA_CLEVELAND_PATH, names=DATA_COLUMNS_NAMES, na_values="?")

    # Drop rows with nan values 
    cleveland_pd = cleveland_pd.dropna().reset_index(drop=True)

    # Conduct Exploratory Data Analysis (EDA) with histograms, correlation heatmaps, and boxplots
    # eda(cleveland_pd)

    # Binary Target Mapping
    X = cleveland_pd.drop(columns=["num"])
    y = (cleveland_pd["num"] > 0).astype(int)

    # Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature Scaling and Encoding Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    # Fit scaler ONLY on X_train to prevent data leakage
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test