import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("student-mat.csv", sep=";")

# Study time vs grade
sns.scatterplot(x=data["studytime"], y=data["G3"])
plt.title("Study Time vs Final Grade")
plt.show()

# Absences vs grade
sns.scatterplot(x=data["absences"], y=data["G3"])
plt.title("Absences vs Final Grade")
plt.show()

# Failures distribution
sns.countplot(x=data["failures"])
plt.title("Past Failures Distribution")
plt.show()