from utils.datasets import generate_data, split_data
from utils.metrics import accuracy_score

from neural_networks.perceptron import SLP


if __name__ == "__main__":
    X, y = generate_data(1000, type="linear")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    clf = SLP()
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Accuracy (train, test): {train_acc}, {test_acc}")