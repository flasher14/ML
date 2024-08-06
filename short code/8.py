import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print("\nCorrect Predictions:")
for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
        print(X_test[i], iris.target_names[y_test[i]], iris.target_names[predictions[i]])
print("\nWrong Predictions:")
for i in range(len(predictions)):
    if predictions[i] != y_test[i]:
        print(X_test[i], iris.target_names[y_test[i]], iris.target_names[predictions[i]])
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)