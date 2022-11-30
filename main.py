import numpy as np

from utils.datasets import generate_data, split_data, plot_data
from support_vector_machines.svm import SVM

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    X, y = generate_data(lin_sep=False)
    X_train, X_test, y_train, y_test = split_data(X, y)
    # plot_data(X, y)
    
    # my implementation
    clf = SVM(kernel="poly")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # sklearn baseline
    clf_baseline = SVC(kernel="poly", degree=2)
    clf_baseline.fit(X_train, y_train)
    y_baseline_pred = clf_baseline.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Accuracy (baseline): {accuracy_score(y_test, y_baseline_pred)}")
    
    # print(clf.b, clf_baseline.intercept_)