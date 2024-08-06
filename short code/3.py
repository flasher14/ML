import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()
removed = [0, 50, 100]
X, y = np.delete(iris.data, removed, axis=0), np.delete(iris.target,removed)
clf = DecisionTreeClassifier().fit(X, y)
predictions = clf.predict(iris.data[removed])
print("Original Labels:", iris.target[removed])
print("Labels Predicted:", predictions)
plot_tree(clf)