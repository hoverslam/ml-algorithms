from sklearn.neural_network import MLPClassifier

from utils.datasets import generate_data, split_data
from neural_network.perceptron import MLP


if __name__ == "__main__":
    X, y = generate_data(10000, type="clusters")
    X_train, X_test, y_train, y_test = split_data(X, y)

    clf1 = MLP(X.shape[1], (100, ), epochs=100)
    clf1.fit(X_train, y_train, plot_history=True)
    train_acc = clf1.score(X_train, y_train)
    test_acc = clf1.score(X_test, y_test)
    print(f"Accuracy [clf1]: {train_acc:.4f} (train), {test_acc:.4f} (test)")

    ## Scikit-learn implementation
    clf2 = MLPClassifier()
    clf2.fit(X_train, y_train)
    train_acc = clf2.score(X_train, y_train)
    test_acc = clf2.score(X_test, y_test)
    print(f"Accuracy [clf2]: {train_acc:.4f} (train), {test_acc:.4f} (test)")