import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[1], [2], [3], [4], [5], [6]]) # X = input feature (hours studied)
y = np.array([0, 0, 0, 1, 1, 1]) # y = output label (0 = Fail, 1 = Pass)

model = LogisticRegression()

model.fit(X, y)

new_data = np.array([[3.5]])

prediction = model.predict(new_data)

probability = model.predict_proba(new_data)

print("Input (Hours Studied):", new_data[0][0])
print("Predicted Class (0=Fail, 1=Pass):", prediction[0])
print("Probability of Fail and Pass:", probability)