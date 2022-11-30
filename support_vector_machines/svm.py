import numpy as np
import cvxopt


class SVM:
    # TODO: bias, predict

    def __init__(self, kernel:str="linear", degree:int=2, gamma:float=0.1) -> None:
        self.degree = degree
        self.gamma = gamma
        
        kernel_functions = {
            "linear": (lambda x1, x2: np.dot(x1, x2)),
            "poly": (lambda x1, x2: (1 + np.dot(x1, x2))**self.degree),
            "rbf": (lambda x1, x2: np.exp(-(np.linalg.norm(x1 - x2)**2) / (2 * self.gamma**2)))
        }
        self.kernel = kernel_functions[kernel]
        
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        n_observations, _ = X.shape
        
        # kernel matrix: unfortunately in O(n^2)
        K = np.zeros((n_observations, n_observations))
        for i in range(n_observations):
            for j in range(n_observations):
                K[i, j] = self.kernel(X[i], X[j])

        # solve quadratic programming problem
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n_observations))
        G = cvxopt.matrix(-np.eye(n_observations))
        h = cvxopt.matrix(np.zeros(n_observations))
        A = cvxopt.matrix(y, (1, n_observations), "d")
        b = cvxopt.matrix(np.zeros(1))
        
        cvxopt.solvers.options["show_progress"] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
 
        # support vectors # TODO: write nicer code
        tol = 1e-5
        mask = np.ravel(solution["x"]) > tol
        idx = np.arange(len(np.ravel(solution["x"])))[mask]
        self.alphas = (np.ravel(solution["x"]))[mask]
        self.sv_X = X[mask]
        self.sv_y = y[mask]
        
        # calulate bias # ! different bias to sklearn SVC
        self.b = 0
        for n in range(len(self.alphas)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.alphas * self.sv_y * K[idx[n], mask])
        self.b /= len(self.alphas)
    
    def predict(self, X:np.ndarray) -> None:
        # TODO: write nicer code
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv_X in zip(self.alphas, self.sv_y, self.sv_X):
                s += a * sv_y * self.kernel(X[i], sv_X)
            y_predict[i] = s

        return np.sign(y_predict + self.b)