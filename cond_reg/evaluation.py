import numpy as np
from typing import Dict

from ..statistical.correlation import Correlation

class Evaluation:

    @staticmethod
    def mae(gt: np.ndarray, pred: np.ndarray) -> float:
        mae_score = np.mean(np.abs(gt - pred))
        return float(mae_score)

    @staticmethod
    def mse(gt: np.ndarray, pred: np.ndarray) -> float:
        mse_score = np.mean(np.power(gt - pred, 2.0))
        return float(mse_score)

    @staticmethod
    def rmse(gt: np.ndarray, pred: np.ndarray) -> float:
        rmse_score = np.sqrt(np.mean(np.power(gt - pred, 2.0)))
        return float(rmse_score)

    @staticmethod
    def r2(gt: np.ndarray, pred: np.ndarray) -> float:
        resid_sum_squares = np.sum(np.power(gt - pred, 2.0))
        tot_sum_squares = np.sum(np.power(gt - np.mean(gt), 2.0))
        if tot_sum_squares != 0.0:
            return float(1.0 - (resid_sum_squares / tot_sum_squares))
        else:
            return 0.0
        
    @staticmethod
    def pearson(gt: np.ndarray, pred: np.ndarray) -> float:
        score = Correlation.pearson(gt, pred)
        return float(score)
    
    @staticmethod
    def spearman(gt: np.ndarray, pred: np.ndarray) -> float:
        score = Correlation.spearman(gt, pred)
        return float(score)

    @staticmethod
    def get_all(gt: np.ndarray, pred: np.ndarray) -> Dict:
        err_dict = {
            "mae": Evaluation.mae(gt, pred),
            "mse": Evaluation.mse(gt, pred),
            "rmse": Evaluation.rmse(gt, pred),
            "r2": Evaluation.r2(gt, pred),
            "pearson": Evaluation.pearson(gt, pred),
            "spearman": Evaluation.spearman(gt, pred)
        }
        return err_dict
