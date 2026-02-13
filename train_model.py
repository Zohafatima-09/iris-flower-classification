from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
X, y = load_iris(return_X_y=True)

# Create scaler and scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "iris_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model saved as iris_model.pkl")
print("✅ Scaler saved as scaler.pkl")
