import pickle

# Load trained model
with open("student_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Enter student details")

studytime = float(input("Study Time (1-4): "))
failures = float(input("Past Failures: "))
absences = float(input("Absences: "))
G1 = float(input("First Period Grade: "))
G2 = float(input("Second Period Grade: "))

# Make prediction
prediction = model.predict([[studytime, failures, absences, G1, G2]])

grade = prediction[0]

print("\nPredicted Final Grade:", round(grade, 2))

# Intervention logic
if grade < 10:
    print("Status: At Risk Student (Needs Intervention)")
elif grade < 14:
    print("Status: Average Performance")
else:
    print("Status: Good Performance")