from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = "secret123"

# Temporary user storage
users = {}

# Load model
model = joblib.load("student_model.pkl")

# Accuracy
accuracy = 0.82


# Chart function
def generate_chart(studytime=None, grade=None):
    data = pd.read_csv("student-mat.csv", sep=";")

    plt.figure(figsize=(8,5))
    plt.scatter(data["studytime"], data["G3"], label="Existing Students")

    # ✅ FIXED CONDITION
    if studytime is not None and grade is not None:
        plt.scatter(studytime, grade, color="red", s=100)

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


# HOME
@app.route("/")
def start():
    return render_template("home.html")


# SIGNUP
@app.route("/signup", methods=["GET", "POST"])
def signup():
    message = None

    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        confirm = request.form["confirm_password"].strip()

        # ✅ CHECK PASSWORD MATCH
        if password != confirm:
            message = "Passwords do not match!"

        # ✅ CHECK DUPLICATE USER
        elif username in users:
            message = "User already exists!"

        else:
            users[username] = password
            return redirect(url_for("login"))

    return render_template("signup.html", message=message)


# LOGIN
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        if username in users and users[username] == password:
            session["user"] = username
            return redirect(url_for("predict"))
        else:
            error = "Invalid username or password"

    return render_template("login.html", error=error)


# LOGOUT
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))   # ✅ FIXED


# PREDICT
@app.route("/predict", methods=["GET", "POST"])
def predict():

    # ✅ ONLY CHECK LOGIN (remove signed_up logic)
    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    status = None
    suggestion = None
    submitted = False

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

        if result < 10:
            status = "⚠️ At Risk"
            suggestion = "Study more and reduce absences"
        elif result < 14:
            status = "📘 Average"
            suggestion = "Improve consistency"
        else:
            status = "✅ Good"
            suggestion = "Keep it up"

        chart = generate_chart(studytime, prediction)

    return render_template(
        "index.html",
        prediction=prediction,
        status=status,
        suggestion=suggestion,
        submitted=submitted,
        chart=chart,
        accuracy=accuracy
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)