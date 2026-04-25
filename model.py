import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
data = pd.read_csv("student-mat.csv", sep=";")

# Remove extreme outliers (optional but useful)
data = data[data["absences"] < 30]

# Features and target
X = data[['studytime','failures','absences','G1','G2','health','freetime','goout']]
y = data['G3']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models (simple + effective)
models = {
    "Decision Tree": DecisionTreeRegressor(max_depth=10),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Linear Regression": LinearRegression()
}

best_model = None
best_error = float("inf")
best_accuracy = 0

print("\n--- Model Comparison ---")

# Train and compare
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    error = mean_absolute_error(y_test, predictions)
    accuracy = r2_score(y_test, predictions)

    print(f"{name} -> MAE: {error:.2f}, R2 Score: {accuracy:.2f}")

    if error < best_error:
        best_error = error
        best_model = model
        best_accuracy = accuracy

# Save best model
joblib.dump(best_model, "student_model.pkl")

print("\n Best model saved successfully")
print(f" Accuracy : {best_accuracy:.2f}")