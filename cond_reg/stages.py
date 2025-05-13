import numpy as np
import pandas as pd

from typing import Tuple, Union, Dict, List
from ...statistical.categorical import ByCategory
from ...statistical.correlation import Correlation
from ...statistical.scaling import Scaling
from .base import CndRegModel


class CndRegStages:

    def __init__(
        self,
        cnd_var_name: str,
        trgt_var_name: str,
        num_stages: int = 1,
        discretize_cnd_var: bool = True
    ):
        self.__cnd_var_name = cnd_var_name
        self.__trgt_var_name = trgt_var_name
        self.__num_stages = num_stages
        self.__discretize_cnd_var = discretize_cnd_var

        self.__expl_var_names = {}
        self.__model = {}

    @property
    def cnd_var_name(self):
        return self.__cnd_var_name

    @property
    def trgt_var_name(self):
        return self.__trgt_var_name
    
    @property
    def has_been_discretized(self):
        return self.__discretize_cnd_var

    def train(
        self,
        df: pd.DataFrame,
        stage2rank: Union[Dict, None] = None,
        max_rank: Union[int, None] = None
    ) -> None:

        previous_model_names = []
        if stage2rank is None:
            stage2rank = {stage: 0 for stage in range(self.__num_stages)}

        prediction = np.zeros(len(df))
        ground_truth = df[self.__trgt_var_name].to_numpy()

        for stage in range(self.__num_stages):

            rank = stage2rank[stage]
            df[self.__trgt_var_name] = df[self.__trgt_var_name].to_numpy() - prediction

            scores, model_names = self.__get_correlation_ranking(df)

            indices = np.argsort(np.array(scores))[::-1]
            if max_rank is not None:
                indices = indices[:max_rank]

            idx = indices[rank]
            selected_model_name = model_names[idx]

            self.__model[stage] = CndRegModel(
                cnd_var_name = self.__cnd_var_name, 
                expl_var_name = selected_model_name, 
                trgt_var_name = self.__trgt_var_name,
                discretize_cnd_var = self.__discretize_cnd_var
            )

            self.__model[stage].train(
                cnd_var_data = df[self.__cnd_var_name].to_numpy(),
                expl_var_data = df[selected_model_name].to_numpy(),
                trgt_var_data = df[self.__trgt_var_name].to_numpy(),
            )

            self.__expl_var_names[stage] = selected_model_name
            previous_model_names.append(selected_model_name)

            prediction = self.__model[stage].predict(
                cnd_var_data = df[self.__cnd_var_name].to_numpy(),
                expl_var_data = df[selected_model_name].to_numpy()
            )

        df[self.__trgt_var_name] = ground_truth


    def predict(
        self,
        df: pd.DataFrame
    ) -> None:
        prediction = np.zeros(len(df))
        for stage in range(self.__num_stages):
            model_name = self.__expl_var_names[stage]
            prediction += self.__model[stage].predict(
                cnd_var_data = df[self.__cnd_var_name].to_numpy(),
                expl_var_data = df[model_name].to_numpy(),
            )
        return prediction


    def __get_correlation_ranking(
        self,
        df: pd.DataFrame
    ) -> Tuple[List, List]:
        scores, model_names = [], []
        trgt_var_arr = df[self.__trgt_var_name].to_numpy()
        cat2idxs = ByCategory.map_categories_to_idxs(arr = df[self.__cnd_var_name].to_numpy())
        for var_name in df.columns:
            if self.__variable_can_be_trained(df, var_name):
                expl_var_arr = df[var_name].to_numpy()
                sub_scores = []
                for _, idxs in cat2idxs.items():
                    sub_scr = Correlation.spearman(expl_var_arr[idxs], trgt_var_arr[idxs])
                    sub_scores.append(np.abs(sub_scr))
                score = sub_scores[0] if len(sub_scores) < 2 else np.median(sub_scores)
                scores.append(score)
                model_names.append(var_name)
        return scores, model_names


    def __variable_can_be_trained(
        self, 
        df: pd.DataFrame, 
        var_name: str
    ) -> bool:
        excl_vars = [self.__cnd_var_name, self.__trgt_var_name]
        valid_types = [np.int32, np.float32, np.int64, np.float64, int, float]
        if df[var_name].dtype in valid_types and var_name not in excl_vars:
            return True
        else:
            return False
