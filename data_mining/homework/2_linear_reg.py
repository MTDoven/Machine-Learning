import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 16

np.random.seed(42)
num_students = 100
study_hours = np.random.uniform(0, 24, num_students)
grades = 5 * study_hours + 50 + np.random.normal(0, 5, num_students)

X = np.vstack([study_hours, np.ones(len(study_hours))]).T
theta = np.linalg.lstsq(X, grades, rcond=None)[0]

a, b = theta
print(f"Model: y = {a:.2f}x + {b:.2f}")

predicted_grades = a * study_hours + b

plt.scatter(study_hours, grades, color='blue', label='data')
plt.plot(study_hours, predicted_grades, color='red', label='model')
plt.xlabel('time (hour)')
plt.ylabel('grade (score)')
plt.legend()
plt.show()
