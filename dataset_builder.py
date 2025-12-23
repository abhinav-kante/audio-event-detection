import os
import pandas as pd
from feature_extraction import extract_features

DATASET_PATH = "data"

features_list = []
labels_list = []

for label in os.listdir(DATASET_PATH):
    class_folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(class_folder):
        continue

    for file in os.listdir(class_folder):
        if file.endswith(".wav"):
            file_path = os.path.join(class_folder, file)

            features = extract_features(file_path)

            if features is not None:
                features_list.append(features)
                labels_list.append(label)

X = pd.DataFrame(features_list)
y = pd.Series(labels_list, name="label")

print("Dataset shape:", X.shape)
print("Labels:")
print(y.value_counts())
