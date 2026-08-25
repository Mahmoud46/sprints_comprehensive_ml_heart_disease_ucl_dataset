DATA_COLUMNS_NAMES = [
    "age",
    "sex", # (1 = male; 0 = female)
    "cp",
    "trestbps", # resting blood pressure (in mm Hg on admission to the hospital)
    "chol", # serum cholestoral in mg/dl
    "fbs", # (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
    "restecg", # (0, 1, 2)
    "thalach", # maximum heart rate achieved
    "exang", # exercise induced angina (1 = yes; 0 = no)
    "oldpeak", # ST depression induced by exercise relative to rest
    "slope", # (1, 2, 3)
    "ca", # number of major vessels (0-3) colored by flourosopy
    "thal", # 3 = normal; 6 = fixed defect; 7 = reversable defect
    "num", # (the predicted attribute) (0 = healthy; 1 => [1, 2, 3, 4] = Disease)
]

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


'''
"age",
"sex": (1 = male; 0 = female)
"cp": chest pain type
        -- Value 1: typical angina
        -- Value 2: atypical angina
        -- Value 3: non-anginal pain
        -- Value 4: asymptomatic,
"trestbps", resting blood pressure (in mm Hg on admission to the hospital)
"chol", serum cholestoral in mg/dl
"fbs", (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
"restecg":resting electrocardiographic results
            -- Value 0: normal
            -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
            -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
"thalach", maximum heart rate achieved
"exang", exercise induced angina (1 = yes; 0 = no)
"oldpeak", ST depression induced by exercise relative to rest
"slope": the slope of the peak exercise ST segment
            -- Value 1: upsloping
            -- Value 2: flat
            -- Value 3: downsloping
"ca": number of major vessels (0-3) colored by flourosopy
"thal": 3 = normal; 6 = fixed defect; 7 = reversable defect
"num": (the predicted attribute) 0 = healthy; 1 => [1, 2, 3, 4] = Disease
'''