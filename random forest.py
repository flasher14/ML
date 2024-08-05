import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Test accuracy: {accuracy}')

sample_index = 8  
new_sample = X_test[sample_index].reshape(1, -1)  
predicted_label = model.predict(new_sample)
actual_label = y_test[sample_index]

print(f'Predicted label: {predicted_label[0]}, Actual label: {actual_label}')

plt.figure(figsize=(4, 4))
plt.imshow(new_sample.reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
plt.title(f'Predicted: {predicted_label[0]}, Actual: {actual_label}')
plt.show()
