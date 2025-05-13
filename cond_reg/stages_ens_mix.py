import numpy as np
import pandas as pd

from typing import Tuple, Dict, List, Union

from .base import CndRegModel
from ...statistical.correlation import Correlation
from ...statistical.categorical import ByCategory
from ...variables.continuous import ContinuousVar


class CndRegStagesEnsembleMix:

    def __init__(
        self,
        config: Dict
    ):
        self.__cnd_var_names = config["conditionals"]
        self.__discretizations = {c: d for c, d in zip(config["conditionals"], config["discretizations"])}

        self.__trgt_var_name = config["target"]
        self.__num_stages = config["stages"]
        self.__num_candidates = config["num_candidates"]

        self.__model = {}

    @property
    def cnd_var_names(self):
        return self.__cnd_var_names

    @property
    def trgt_var_name(self):
        return self.__trgt_var_name

    def train(
        self,
        df: pd.DataFrame
    ) -> None:
        
        df_train = df.copy()

        prediction = np.zeros(len(df_train))
        ground_truth = df_train[self.__trgt_var_name].to_numpy()
        current_ground_truth = np.copy(ground_truth)
        initial_model_names = self.__model_selection(df_train, ground_truth = ground_truth)

        intermediate_predictions = {}
        intermediate_ground_truths = {}
        for num_models in range(self.__num_candidates):
            self.__model[num_models] = {stage: None for stage in range(self.__num_stages)}
            intermediate_predictions[num_models] = {stage: None for stage in range(self.__num_stages)}
            intermediate_ground_truths[num_models] = {stage: None for stage in range(self.__num_stages)}

        for model_num, (cnd_var_name, expl_var_name) in enumerate(initial_model_names):

            self.__model[model_num][0] = CndRegModel(
                cnd_var_name = cnd_var_name,
                expl_var_name = expl_var_name,
                trgt_var_name = self.__trgt_var_name,
                discretize_cnd_var = self.__discretizations[cnd_var_name]
            )

            self.__model[model_num][0].train(
                cnd_var_data = df_train[cnd_var_name].to_numpy(),
                expl_var_data = df_train[expl_var_name].to_numpy(),
                trgt_var_data = current_ground_truth
            )

            prediction = self.__model[model_num][0].predict(
                cnd_var_data = df_train[cnd_var_name].to_numpy(),
                expl_var_data = df_train[expl_var_name].to_numpy()
            )

            intermediate_predictions[model_num][0] = prediction
            intermediate_ground_truths[model_num][0] = self.__absolute_difference(current_ground_truth, prediction)

        for model_num in range(self.__num_candidates):
            for stage in range(1, self.__num_stages):

                current_ground_truth = intermediate_ground_truths[model_num][stage - 1]
                selected_model_names = self.__model_selection(df_train, ground_truth = current_ground_truth)
                cnd_var_name, expl_var_name = selected_model_names[0]

                self.__model[model_num][stage] = CndRegModel(
                    cnd_var_name = cnd_var_name,
                    expl_var_name = expl_var_name,
                    trgt_var_name = self.__trgt_var_name,
                    discretize_cnd_var = self.__discretizations[cnd_var_name]
                )

                self.__model[model_num][stage].train(
                    cnd_var_data = df_train[cnd_var_name].to_numpy(),
                    expl_var_data = df_train[expl_var_name].to_numpy(),
                    trgt_var_data = current_ground_truth
                )

                prediction = self.__model[model_num][stage].predict(
                    cnd_var_data = df_train[cnd_var_name].to_numpy(),
                    expl_var_data = df_train[expl_var_name].to_numpy()
                )

                intermediate_predictions[model_num][stage] = prediction + intermediate_predictions[model_num][stage - 1]
                intermediate_ground_truths[model_num][stage] = self.__absolute_difference(current_ground_truth, prediction)

    def predict(
        self,
        df: pd.DataFrame
    ) -> np.ndarray:
        predictions = self.predict_all(df)
        return np.mean(predictions, axis = 0)

    def predict_all(
        self,
        df: pd.DataFrame
    ) -> List:

        df_pred = df.copy()

        predictions = []
        for model_num in range(self.__num_candidates):

            single_model_prediction = np.zeros(len(df_pred))
            for stage in range(self.__num_stages):

                cnd_var_name = self.__model[model_num][stage].cnd_var_name
                expl_var_name = self.__model[model_num][stage].expl_var_name

                single_model_prediction += self.__model[model_num][stage].predict(
                    cnd_var_data = df_pred[cnd_var_name].to_numpy(),
                    expl_var_data = df_pred[expl_var_name].to_numpy()
                )

            predictions.append(single_model_prediction) 

        return predictions

    def __model_selection(
        self, 
        df: pd.DataFrame,
        ground_truth: np.ndarray
    ) -> List:
        scores, model_names = [], []
        for cnd_var_name in self.__cnd_var_names:
            scrs, mdls = self.__get_correlation_ranking(df, cnd_var_name, ground_truth)
            model_names += mdls
            scores += scrs
        idxs = np.argsort(scores)[::-1]
        scores = np.array(scores)[idxs]
        model_names = np.array(model_names)[idxs]
        return model_names[:self.__num_candidates]

    def __get_correlation_ranking(
        self,
        df: pd.DataFrame,
        cnd_var_name: str,
        ground_truth: np.ndarray
    ) -> Tuple[List, List]:
        scores, model_names = [], []
        cat2idxs = ByCategory.map_categories_to_idxs(arr = df[cnd_var_name].to_numpy())
        for var_name in df.columns:
            if self.__variable_can_be_trained(df, var_name):
                sub_scores = []
                expl_var_arr = df[var_name].to_numpy()
                for _, idxs in cat2idxs.items():
                    sub_scr = Correlation.spearman(expl_var_arr[idxs], ground_truth[idxs])
                    sub_scores.append(np.abs(sub_scr))
                score = sub_scores[0] if len(sub_scores) < 2 else np.median(sub_scores)
                scores.append(score)
                model_names.append((cnd_var_name, var_name))
        return scores, model_names

    def __get_correlation_ranking_(
        self,
        df: pd.DataFrame,
        cnd_var_name: str,
        ground_truth: np.ndarray,
        preselected_variables: Union[List, None] = None
    ) -> Tuple[List, List]:

        scores, model_names = [], []
        var0 = ContinuousVar(arr = ground_truth, auto_discretization = True)
        cat2idxs = ByCategory.map_categories_to_idxs(arr = df[cnd_var_name].to_numpy())
        
        preselected_variables = list(df.columns) if preselected_variables is None else preselected_variables

        for var_name in preselected_variables:
            if self.__variable_can_be_trained(df, var_name):

                sub_scores = []
                expl_var_arr = df[var_name].to_numpy()

                var1 = ContinuousVar(arr = expl_var_arr, auto_discretization = True)
                var1.data_discrete

                for _, idxs in cat2idxs.items():
                    sub_scr = Correlation.mi(var0.data_discrete[idxs], var1.data_discrete[idxs])
                    sub_scores.append(np.abs(sub_scr))
                score = sub_scores[0] if len(sub_scores) < 2 else np.median(sub_scores)
                scores.append(score)
                model_names.append((cnd_var_name, var_name))

        return scores, model_names

    def __variable_can_be_trained(
        self, 
        df: pd.DataFrame, 
        var_name: str
    ) -> bool:
        valid_types = [np.int32, np.float32, np.int64, np.float64, int, float]
        if df[var_name].dtype in valid_types and var_name != self.__trgt_var_name:
            return True
        else:
            return False

    def __absolute_difference(
        self,
        ground_truth: np.ndarray,
        prediction: np.ndarray
    ) -> np.ndarray:
        return ground_truth - prediction

    @staticmethod
    def get_correlation_ranking_beta(
        df: pd.DataFrame,
        cnd_var_name: str,
        ground_truth: np.ndarray,
        trgt_var_name: str
    ) -> Tuple[List, List]:
        
        var0 = ContinuousVar(arr = ground_truth, auto_discretization = True)

        scores, model_names = [], []
        cat2idxs = ByCategory.map_categories_to_idxs(arr = df[cnd_var_name].to_numpy())
        for var_name in df.columns:
            valid_types = [np.int32, np.float32, np.int64, np.float64, int, float]
            if df[var_name].dtype in valid_types and var_name != trgt_var_name:

                sub_scores = []
                expl_var_arr = df[var_name].to_numpy()

                var1 = ContinuousVar(arr = expl_var_arr, auto_discretization = True)
                var1.data_discrete

                for _, idxs in cat2idxs.items():
                    sub_scr = Correlation.mi(var0.data_discrete[idxs], var1.data_discrete[idxs])
                    sub_scores.append(np.abs(sub_scr))
                score = sub_scores[0] if len(sub_scores) < 2 else np.median(sub_scores)
                scores.append(score)
                model_names.append((cnd_var_name, var_name))

        idxs = np.argsort(scores)[::-1]
        scores = np.array(scores)[idxs]
        model_names = np.array(model_names)[idxs]

        return scores, model_names
