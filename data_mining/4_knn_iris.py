from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
for i, color in zip(range(len(iris.target_names)), ['red', 'green', 'blue']):
    plt.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1], color=color, label=f"True {iris.target_names[i]}")
    plt.scatter(X_test[y_pred == i, 0], X_test[y_pred == i, 1], color=color, marker='x', label=f"Pred {iris.target_names[i]}")

neighbors = knn.kneighbors(X_test, return_distance=False)
for idx, neighbor_indices in enumerate(neighbors):
    for neighbor_idx in neighbor_indices:
        plt.scatter(X_train[neighbor_idx, 0], X_train[neighbor_idx, 1],
                    color=plt.cm.Paired(y_test[idx] / len(iris.target_names)),
                    s=20, alpha=0.6)
        plt.plot([X_test[idx, 0], X_train[neighbor_idx, 0]],
                 [X_test[idx, 1], X_train[neighbor_idx, 1]],
                 color=plt.cm.Paired(y_test[idx] / len(iris.target_names)),
                 alpha=0.6, linestyle='--')

plt.title("KNN Classification Results with Neighbors")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
