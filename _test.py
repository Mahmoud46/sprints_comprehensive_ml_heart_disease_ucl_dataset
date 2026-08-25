'''
thal   ca   cp  oldpeak  thalach  slope  exang  sex
3.0  0.0  2.0      0.0    182.0    1.0    0.0  1.0
'''

test_X = [[3.0, 0.0, 2.0, 0.0, 182.0, 1.0, 0.0, 1.0], # 0
          [7.0, 2.0, 3.0, 3.2, 173.0, 1.0, 0.0, 1.0] # 1
          ]

import joblib
import numpy as np
from config.paths import FINAL_MODEL_PATH

# Load saved model back into memory
loaded_model = joblib.load(FINAL_MODEL_PATH)

# 2. Input the preprocessed feature sample
# (Reshaped into a 2D array: 1 sample with 8 features)
sample_data = np.array([test_X[1]])

# 3. Generate predictions
predicted_class = loaded_model.predict(sample_data)[0]

# 4. Extract confidence/probability scores (if available)
if hasattr(loaded_model, "predict_proba"):
    probabilities = loaded_model.predict_proba(sample_data)[0]
    class_0_prob = probabilities[0]
    class_1_prob = probabilities[1]

    print(f"Predicted Class: {predicted_class}")
    print(f"Probability Class 0 (Healthy): {class_0_prob * 100:.2f}%")
    print(f"Probability Class 1 (Disease): {class_1_prob * 100:.2f}%")
else:
    print(f"Predicted Class: {predicted_class}")