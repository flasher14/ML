import numpy as np
import matplotlib.pyplot as plt
def lw_reg(xq, X, y, tau):
    W = np.exp(-np.sum((X - xq) ** 2, axis=1) / (2 * tau ** 2))
    return xq @ np.linalg.inv(X.T @ np.diag(W) @ X) @ (X.T @ (W * y))
np.random.seed(0)
X_train = np.linspace(0, 10, 50)
y_train = np.sin(X_train) + np.random.normal(0, 0.1, X_train.size)
predictions = [lw_reg([1, xq], np.c_[np.ones(X_train.size), X_train], y_train, 0.5) for xq in np.linspace(0,10, 100)]

plt.scatter(X_train, y_train, color='blue')
plt.plot(np.linspace(0, 10, 100), predictions, color='red')
plt.xlabel('X'); 
plt.ylabel('Y'); 
plt.title('Locally Weighted Regression'); 
plt.grid(True); 
plt.show()