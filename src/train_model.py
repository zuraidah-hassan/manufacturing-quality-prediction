import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# Setup
# ===============================
os.makedirs("outputs", exist_ok=True)

# ===============================
# Load dataset
# ===============================
data = pd.read_csv("data/raw/manufacturing_data.csv")

X = data.drop("quality", axis=1)
y = data["quality"]

# ===============================
# Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Train model
# ===============================
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ===============================
# Prediction
# ===============================
y_pred = model.predict(X_test)

# ===============================
# Evaluation (text)
# ===============================
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# Confusion Matrix (SAVE IMAGE)
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# ===============================
# Feature Importance (SAVE IMAGE)
# ===============================
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")

plt.savefig("outputs/feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ Evaluation plots saved in /outputs folder")
