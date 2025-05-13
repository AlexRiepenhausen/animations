import numpy as np
from typing import Dict
from collections import defaultdict


class ByCategory:

    @staticmethod
    def map_categories_to_idxs(arr: np.ndarray) -> Dict:
        cat_to_idxs = defaultdict(list)
        for i, cat in enumerate(arr):
            cat_to_idxs[cat].append(i)
        for cat, idxs in cat_to_idxs.items():
            cat_to_idxs[cat] = np.array(idxs).astype(np.int32)
        return cat_to_idxs
    
    @staticmethod
    def split_first_var_by_unique_items_second_var(arr0: np.ndarray, arr1: np.ndarray) -> Dict:
        cat_to_data = {}
        cat_to_idxs = ByCategory.map_categories_to_idxs(arr=arr1)
        for c, idxs in cat_to_idxs.items():
            cat_to_data[c] = arr0[idxs]
        return cat_to_data

    @staticmethod
    def apply_function_by_categories(
        arr_trgt: np.ndarray,
        arr_cat: np.ndarray,
        func,
        **kwargs
    ) -> np.ndarray:
        cat_to_idxs = ByCategory.map_categories_to_idxs(arr_cat)
        arr_new = np.full(arr_cat.shape[0], fill_value=np.nan)
        for _, idxs in cat_to_idxs.items():
            arr_new[idxs] = func(arr_trgt[idxs], **kwargs) if kwargs else func(arr_trgt[idxs])
        return arr_new
