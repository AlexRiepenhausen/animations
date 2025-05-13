import numpy as np

class Scaling:

    @staticmethod
    def min_max(arr: np.ndarray) -> np.ndarray:
        min_v, max_v = np.min(arr), np.max(arr)
        if max_v - min_v > 0.0:
            return (arr - min_v) / (max_v - min_v)
        else:
            return arr

    @staticmethod
    def min_max_given(
        arr: np.ndarray,
        min_v: np.float32, 
        max_v: np.float32
    ) -> np.ndarray:
        if max_v - min_v > 0.0:
            return (arr - min_v) / (max_v - min_v)
        else:
            return arr

    @staticmethod
    def min_max_reverse(
        arr: np.ndarray,
        min_v: np.float32, 
        max_v: np.float32
    ) -> np.ndarray:
        if max_v - min_v > 0.0:
            return arr * (max_v - min_v) + min_v
        else:
            return arr

    @staticmethod
    def zscore(arr: np.ndarray) -> np.ndarray:
        arr_unique, _ = np.unique(arr, return_counts=True)
        unique_item_num = arr_unique.shape[0]
        mean_v, std_v = np.mean(arr), np.std(arr)
        if unique_item_num > 1 and std_v > 0.0:
            return (arr - mean_v) / std_v
        else:
            return arr

    @staticmethod
    def zscore_given(
        arr: np.ndarray,
        mean_v: np.float32,
        std_v: np.float32
    ) -> np.ndarray:
        if std_v > 0.0:
            return (arr - mean_v) / std_v
        else:
            return arr

    @staticmethod
    def zscore_reverse(
        arr: np.ndarray,
        mean_v: np.float32,
        std_v: np.float32
    ) -> np.ndarray:
        if std_v > 0.0:
            return arr * std_v + mean_v
        else:
            return arr

    @staticmethod
    def robust(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] < 3:
            return arr
        iqr = np.percentile(arr, [75])[0] - np.percentile(arr, [25])[0]
        if iqr == 0.0:
            return arr
        return (arr - np.median(arr)) / iqr

    @staticmethod
    def robust_given(
        arr: np.ndarray,
        median_v: np.float32,
        iqr: np.float32
    ) -> np.ndarray:
        if iqr > 0.0:
            return (arr - median_v) / iqr
        else:
            return arr

    @staticmethod
    def robust_reverse(
        arr: np.ndarray,
        median_v: np.float32,
        iqr: np.float32
    ) -> np.ndarray:
        if iqr > 0.0:
            return arr * iqr + median_v
        else:
            return arr

    @staticmethod
    def log10(arr: np.ndarray) -> np.ndarray:
        return np.log10(1.0 + np.abs(arr)) * np.sign(arr)

    @staticmethod
    def log10_reverse(arr: np.ndarray) -> np.ndarray:
        return (np.power(10, np.abs(arr)) - 1.0) * np.sign(arr)
