from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Load trained model
model = joblib.load("student_model.pkl")


# Generate chart function
def generate_chart(studytime=None, grade=None):

    data = pd.read_csv("student-mat.csv", sep=";")

    plt.figure()

    # Existing dataset students
    plt.scatter(data["studytime"], data["G3"], label="Existing Students")

    # New predicted student point
    if studytime is not None and grade is not None:
        plt.scatter(studytime, grade, color="red", s=100, label="Predicted Student")

    plt.xlabel("Study Time")
    plt.ylabel("Final Grade")
    plt.title("Study Time vs Final Grade")

    plt.legend()

    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)

    chart = base64.b64encode(img.getvalue()).decode()

    plt.close()

    return chart


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    status = None
    submitted = False

    # Default chart
    chart = generate_chart()

    if request.method == "POST":

        submitted = True

        studytime = float(request.form["studytime"])
        failures = float(request.form["failures"])
        absences = float(request.form["absences"])
        G1 = float(request.form["G1"])
        G2 = float(request.form["G2"])

        features = np.array([[studytime, failures, absences, G1, G2]])

        result = model.predict(features)[0]

        prediction = round(result, 2)

        # Risk detection
        if result < 10:
            status = "Student At Risk⚠️"
        elif result < 14:
            status = "Average Performance"
        else:
            status = "Good Performance"

        # Update chart with entered student
        chart = generate_chart(studytime, prediction)

    return render_template(
        "index.html",
        prediction=prediction,
        status=status,
        submitted=submitted,
        chart=chart
    )


if __name__ == "__main__":
    app.run(debug=True)