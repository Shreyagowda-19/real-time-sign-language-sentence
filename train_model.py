import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

DATA_DIR = "data"
labels = []
features = []

label_map = {
    "Hello": 0,
    "Yes": 1,
    "No": 2,
    "Thanks": 3
}

# Load data
for label_name, label_value in label_map.items():
    folder_path = os.path.join(DATA_DIR, label_name)

    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            file_path = os.path.join(folder_path, file)
            data = np.load(file_path)

            features.append(data)
            labels.append(label_value)

X = np.array(features)
y = np.array(labels)

print("Total samples:", X.shape[0])
print("Feature size:", X.shape[1])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model accuracy:", accuracy)

# Save model
with open("sign_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as sign_model.pkl")
