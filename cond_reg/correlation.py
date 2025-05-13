import numpy as np
from .encoding import Encoding
from sklearn.metrics import adjusted_mutual_info_score

class Correlation:

    @staticmethod
    def pearson(x: np.ndarray, y: np.ndarray) -> np.float32:
        x_std, y_std = np.std(x), np.std(y)
        if x_std > 0.0 and y_std > 0.0:
            return np.corrcoef(x, y)[0][1]
        else:
            return 0.0

    @staticmethod
    def spearman(x: np.ndarray, y: np.ndarray) -> np.float32:
        x_std, y_std = np.std(x), np.std(y)
        if x_std > 0.0 and y_std > 0.0:
            x_rank = x.argsort().argsort().astype(np.float64)
            y_rank = y.argsort().argsort().astype(np.float64)
            d_rank = np.power(x_rank - y_rank, 2.0)
            nitems = x.shape[0]
            return 1.0 - (6.0 * np.sum(d_rank) / (np.power(nitems, 3.0) - nitems))
        else:
            return 0.0

    @staticmethod
    def dot_product(x: np.ndarray, y: np.ndarray) -> np.float32:
        return np.abs(np.dot(x, y))
    
    @staticmethod
    def xicor(x: np.ndarray, y: np.ndarray) -> np.float32:
        x_std, y_std = np.std(x), np.std(y)
        if x_std > 0.0 and y_std > 0.0:
            n, order = x.shape[0], np.argsort(x)
            r = np.argsort(np.argsort(y[order])) + 1
            d = np.sum(np.abs(r[1:] - r[:n-1]))
            return 1.0 - 3.0 * d / ((n ** 2) - 1)
        else:
            return 0.0

    @staticmethod
    def mi(x: np.ndarray, y: np.ndarray) -> np.float32:
        return adjusted_mutual_info_score(x, y)
