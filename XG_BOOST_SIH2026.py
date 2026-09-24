
# SIH 2026 - Multispectral Sensor XGBoost Model


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier
import joblib





SENSOR_CHANNELS = [

    "B1x",
    "B1y",
    "B1z",

    "B2x",
    "B2y",
    "B2z",

    "dBx",
    "dBy",
    "dBz",

    "magnetic_magnitude",

    "acoustic_TOF",
    "acoustic_amplitude",
    "acoustic_backscatter",

    "CSEM_voltage",
    "CSEM_current",
    "electric_field",

    "pressure",
    "temperature",

    "ax",
    "ay",
    "az",

    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z"
]




N_SAMPLES = 1000
SPECTRUM_LENGTH = 256

RANDOM_STATE = 42




print("\nGenerating demonstration dataset...")

np.random.seed(RANDOM_STATE)

X_raw = np.random.randn(
    N_SAMPLES,
    len(SENSOR_CHANNELS),
    SPECTRUM_LENGTH
)




y = np.random.randint(
    0,
    3,
    N_SAMPLES
)

print("Raw input shape:")
print(X_raw.shape)

print("\nTarget shape:")
print(y.shape)



def extract_features(X):
    """
    Convert each spectrum into statistical/spectral features.

    Input:
        X = samples × channels × spectrum_length

    Output:
        feature matrix
    """

    features = []

    for sample in X:

        sample_features = []

        for channel in sample:

            

            mean_value = np.mean(channel)

            std_value = np.std(channel)

            min_value = np.min(channel)

            max_value = np.max(channel)

            rms_value = np.sqrt(
                np.mean(channel ** 2)
            )

            median_value = np.median(channel)

            energy = np.sum(channel ** 2)

            

            peak_index = np.argmax(
                np.abs(channel)
            )

            peak_value = channel[peak_index]

            
            magnitude = np.abs(channel)

            frequencies = np.arange(
                len(channel)
            )

            if np.sum(magnitude) != 0:

                spectral_centroid = (
                    np.sum(
                        frequencies * magnitude
                    )
                    /
                    np.sum(magnitude)
                )

            else:

                spectral_centroid = 0

            

            sample_features.extend([
                mean_value,
                std_value,
                min_value,
                max_value,
                rms_value,
                median_value,
                energy,
                peak_value,
                peak_index,
                spectral_centroid
            ])

        features.append(sample_features)

    return np.array(features)




print("\nExtracting spectral features...")

X_features = extract_features(X_raw)

print("Feature matrix shape:")
print(X_features.shape)



X_train, X_test, y_train, y_test = train_test_split(

    X_features,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


print("\nTraining samples:")
print(X_train.shape[0])

print("Testing samples:")
print(X_test.shape[0])




model = XGBClassifier(

    n_estimators=300,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="multi:softprob",

    num_class=3,

    eval_metric="mlogloss",

    random_state=RANDOM_STATE
)



print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")




y_pred = model.predict(X_test)



accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)



print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)




cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)




importance = model.feature_importances_

indices = np.argsort(
    importance
)[-20:]


plt.figure(
    figsize=(10, 6)
)

plt.barh(
    range(len(indices)),
    importance[indices]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature Index"
)

plt.title(
    "XGBoost Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "results_feature_importance.png",
    dpi=300
)

plt.show()



model.save_model(
    "xgboost_sih_model.json"
)

print(
    "\nModel saved as:"
    " xgboost_sih_model.json"
)



results = pd.DataFrame({

    "Actual": y_test,

    "Predicted": y_pred

})

results.to_csv(
    "xgboost_predictions.csv",
    index=False
)

print(
    "Predictions saved as:"
    " xgboost_predictions.csv"
)


print("\n====================================")
print("XGBOOST PIPELINE COMPLETED")
print("====================================")