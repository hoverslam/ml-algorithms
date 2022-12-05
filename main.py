from sklearn.neural_network import MLPClassifier

from utils.datasets import generate_data, split_data
from neural_network.perceptron import MLP


if __name__ == "__main__":
    X, y = generate_data(1000, type="clusters")
    X_train, X_test, y_train, y_test = split_data(X, y)

    clf = MLP(X.shape[1], (100,), epochs=500)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Accuracy (train, test): {train_acc}, {test_acc}")

    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Accuracy (train, test): {train_acc}, {test_acc}")