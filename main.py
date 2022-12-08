from sklearn import linear_model

from utils.datasets import generate_data, split_data
from linear_model.regression import LinearRegression


if __name__ == "__main__":
    X, y = generate_data(10000, type="regression")
    X_train, X_test, y_train, y_test = split_data(X, y)

    reg1 = LinearRegression()
    reg1.fit(X_train, y_train)
    train_rmse = reg1.score(X_train, y_train)
    test_rmse = reg1.score(X_test, y_test)
    print(f"R2 score [reg1]: {train_rmse:.4f} (train), {test_rmse:.4f} (test)")
    
    # Scikit-learn implementation (baseline)
    reg2 = linear_model.LinearRegression()
    reg2.fit(X_train, y_train)
    train_rmse = reg2.score(X_train, y_train)
    test_rmse = reg2.score(X_test, y_test)
    print(f"R2 score [reg2]: {train_rmse:.4f} (train), {test_rmse:.4f} (test)")