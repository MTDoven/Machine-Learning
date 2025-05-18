import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    return X, y

def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_svm(X_train, y_train, kernel='linear', C=1.0):
    model = SVC(kernel=kernel, C=C, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    svm_model = train_svm(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test)

    import matplotlib.pyplot as plt
    def plot_svm_decision_boundary(model, X, y):
        plt.figure(figsize=(8, 6))
        for i, color in zip(range(len(set(y))), ['red', 'green', 'blue']):
            plt.scatter(X[y == i, 0], X[y == i, 1], color=color, label=f"Class {i}")

        plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                    s=100, facecolors='none', edgecolors='k', label='Support Vectors')

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)

        plt.title("SVM Decision Boundary with Support Vectors")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

    plot_svm_decision_boundary(svm_model, X, y)
