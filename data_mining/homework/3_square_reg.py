import numpy as np
import matplotlib.pyplot as plt

time = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
viral_load = np.array([70, 500, 3200, 8000, 15000, 20000, 15000, 8000, 3200, 500])

X = np.vstack([time**2, time, np.ones(len(time))]).T
y = viral_load

theta = np.linalg.lstsq(X, y, rcond=None)[0]
a, b, c = theta

print(f"Model: y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}")

time_pred = np.linspace(0, 30, 60)
viral_load_pred = a * time_pred**2 + b * time_pred + c

plt.scatter(time, viral_load, color='blue', label='data')
plt.plot(time_pred, viral_load_pred, color='red', label='model')
plt.xlabel('time (hour)')
plt.ylabel('number (copies/ml)')
plt.legend()
plt.tight_layout()
plt.show()
