import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("Student_Performance/StudentsPerformance.csv")
data = pd.get_dummies(data)

X = data[["reading score", "writing score"]]
y = data["math score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print("Accuracy:", score)

plt.scatter(y_test, predictions)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Scores")
plt.show()

print("\n--- Predict Math Score ---")

reading = float(input("Enter reading score: "))
writing = float(input("Enter writing score: "))

input_data = pd.DataFrame([[reading, writing]], columns=["reading score", "writing score"])

prediction = model.predict(input_data)

print("Predicted Math Score:", prediction[0])