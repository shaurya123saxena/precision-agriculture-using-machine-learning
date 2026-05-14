import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Split input and output
X = df.drop('label', axis=1)
y = df['label']

# Split into train/test (VERY IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Better model (THIS IS THE KEY CHANGE 🔥)
model = RandomForestClassifier(
    n_estimators=300,     # more trees
    max_depth=12,         # control overfitting
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model
pickle.dump(model, open("models/new_model.pkl", "wb"))

print("✅ Model trained and saved!")