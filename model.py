import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
data = pd.read_csv("student-mat.csv", sep=";")

# Features and target
X = data[['studytime','failures','absences','G1','G2']]
y = data['G3']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Linear Regression": LinearRegression()
}

best_model = None
best_error = float("inf")

# Train models and compare
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    error = mean_absolute_error(y_test, predictions)

    print(name, "Error:", error)

    if error < best_error:
        best_error = error
        best_model = model

# Save best model
joblib.dump(best_model, "student_model.pkl")

print("Best model saved successfully")