import numpy as np

from typing import Dict
from sklearn.linear_model import RidgeCV

from ..evaluation import Evaluation
from ...statistical.categorical import ByCategory

from ...variables.continuous import ContinuousVar


class CndRegModel:

    def __init__(
        self,
        cnd_var_name: str,
        expl_var_name: str,
        trgt_var_name: str,
        discretize_cnd_var: bool = False
    ):
        self.__cnd_var_name = cnd_var_name
        self.__expl_var_name = expl_var_name
        self.__trgt_var_name = trgt_var_name

        self.__discretize_cnd_var = discretize_cnd_var
        self.__cnd_var_cntnr = None

        self.__aux = {}
        self.__model = {}
        self.__losses = {}

    @property
    def cnd_var_name(self):
        return self.__cnd_var_name

    @property
    def expl_var_name(self):
        return self.__expl_var_name
    
    @property
    def trgt_var_name(self):
        return self.__trgt_var_name
    
    @property
    def has_been_discretized(self):
        return self.__discretize_cnd_var

    def train(
        self,
        cnd_var_data: np.ndarray,
        expl_var_data: np.ndarray,
        trgt_var_data: np.ndarray
    ) -> None:

        cnd_var_data_dscr = np.copy(cnd_var_data)
        if self.__discretize_cnd_var:
            self.__cnd_var_cntnr = ContinuousVar(arr = cnd_var_data, auto_discretization = True)
            cnd_var_data_dscr = self.__cnd_var_cntnr.discretize(arr = cnd_var_data_dscr)

        aux_model = RidgeCV(fit_intercept = True)
        aux_model.fit(expl_var_data.reshape(-1, 1), trgt_var_data.reshape(-1, 1))

        self.__aux = {
            "w": aux_model.coef_[0][0],
            "b": aux_model.intercept_[0],
            "min_trgt": float(np.min(trgt_var_data)),
            "max_trgt": float(np.max(trgt_var_data)),
            "min_expl": float(np.min(expl_var_data)),
            "max_expl": float(np.max(expl_var_data))
        }

        cat2idxs = ByCategory.map_categories_to_idxs(arr = cnd_var_data_dscr)

        for c, idxs in cat2idxs.items():

            model = RidgeCV(fit_intercept = True)
            model.fit(expl_var_data[idxs].reshape(-1, 1), trgt_var_data[idxs].reshape(-1, 1))

            self.__model[c] = {
                "w": model.coef_[0][0],
                "b": model.intercept_[0],
                "min_trgt": float(np.min(trgt_var_data[idxs])),
                "max_trgt": float(np.max(trgt_var_data[idxs])),
                "min_expl": float(np.min(expl_var_data[idxs])),
                "max_expl": float(np.max(expl_var_data[idxs]))
            }

            prediction = self.predict(
                cnd_var_data = cnd_var_data[idxs],
                expl_var_data = expl_var_data[idxs]
            )

            self.__losses[c] = Evaluation.get_all(
                gt = trgt_var_data[idxs],
                pred = prediction
            )

    def predict(
        self,
        cnd_var_data: np.ndarray,
        expl_var_data: np.ndarray
    ) -> np.ndarray:
        
        cnd_var_data_dscr = np.copy(cnd_var_data)
        if self.__discretize_cnd_var:
            cnd_var_data_dscr = self.__cnd_var_cntnr.discretize(arr = cnd_var_data)

        prediction = np.zeros(expl_var_data.shape[0])
        for i, (c, x) in enumerate(zip(cnd_var_data_dscr, expl_var_data)):
            if c in self.__model:
                prediction[i] = self.__single_prediction(
                    w = self.__model[c]["w"],
                    x = x,
                    b = self.__model[c]["b"],
                    trgt_min_val = self.__model[c]["min_trgt"],
                    trgt_max_val = self.__model[c]["max_trgt"]
                )
            else:
                prediction[i] = self.__single_prediction(
                    w = self.__aux["w"],
                    x = x,
                    b = self.__aux["b"],
                    trgt_min_val = self.__aux["min_trgt"],
                    trgt_max_val = self.__aux["max_trgt"]
                )
        return prediction

    def get_train_loss(
        self, 
        cnd_item: object,
        loss_type: str = "rmse"
    ) -> float:
        return self.__losses[cnd_item][loss_type]

    def get_single_model_parameters(
        self,
        cnd_item: object,
    ) -> Dict:
        if cnd_item in self.__model:
            return self.__model[cnd_item]
        else:
            return self.__aux

    def __single_prediction(
        self,
        w: np.float32,
        x: np.float32,
        b: np.float32,
        trgt_min_val: np.float32,
        trgt_max_val: np.float32,
    ) -> float:
        pred_val = w * x + b
        if pred_val > trgt_max_val:
            return trgt_max_val + np.sqrt(1.0 + np.abs(pred_val - trgt_max_val)) - 1.0
        elif pred_val < trgt_min_val:
            return trgt_min_val - np.sqrt(1.0 + np.abs(pred_val - trgt_min_val)) + 1.0
        else:
            return pred_val
