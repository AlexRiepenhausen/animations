import numpy as np
from enum import Enum
from typing import Dict
from ..statistical.scaling import Scaling
from ..statistical.encoding import Encoding

class ScalingType(Enum):

    min_max = 1
    zscore = 2
    robust = 3
    log = 4

    def scale(
        self, 
        arr: np.ndarray
    ) -> np.ndarray:
        if self.name == "min_max":
            return Scaling.min_max(arr)
        if self.name == "zscore":
            return Scaling.zscore(arr)
        if self.name == "robust":
            return Scaling.robust(arr)
        if self.name == "log":
            return Scaling.log10(arr)
        return arr

    def scale_reverse(
        self, 
        arr: np.ndarray,
        min_v: np.float32,
        max_v = np.float32,
        mean_v = np.float32,
        std_v = np.float32,
        median_v = np.float32,
        iqr = np.float32
    ) -> np.ndarray:
        if self.name == "min_max":
            return Scaling.min_max_reverse(arr, min_v = min_v, max_v = max_v)
        if self.name == "zscore":
            return Scaling.zscore_reverse(arr, mean_v = mean_v, std_v = std_v)
        if self.name == "robust":
            return Scaling.robust_reverse(arr, median_v= median_v, iqr = iqr)
        if self.name == "log":
            return Scaling.log10_reverse(arr)
        return arr


class Discretizer:

    def __init__(
        self, 
        arr: np.ndarray,
        auto: bool = False,
        segment_size: int = 48,
        num_segments: int = 4
    ):
        self.__seg2idxs = self.__init_seg2idxs(arr, segment_size = segment_size, num_segments = num_segments, auto = auto)
        self.__bins = Encoding.get_bins_from_segments_to_idxs(arr, self.__seg2idxs)

    @property
    def seg2idxs(self):
        return self.__seg2idxs

    def discretize(self, arr: np.ndarray) -> np.ndarray:
        return Encoding.discretize(arr, bins = self.__bins)

    def __init_seg2idxs(
        self, 
        arr: np.ndarray,
        segment_size: int,
        num_segments: int,
        auto: bool = False
    ) -> Dict:
        
        segment_size = segment_size if segment_size >= 1 else 1
        num_segments = num_segments if num_segments >= 1 else 1

        if auto:
            return Encoding.get_cont_var_segments_to_idxs(arr, segment_size = segment_size)
        else:
            bins = Encoding.get_bins_from_range(arr, num_segments = num_segments)
            arr_discr = Encoding.discretize(arr, bins = bins)
            seg2idxs = Encoding.get_cat_var_segments_to_idxs(arr_discr)
            seg2idxs = {int(k): v for k, v in seg2idxs.items()}
            return seg2idxs


class ContinuousVar:

    def __init__(
        self, 
        arr: np.ndarray,
        scaling: str = "min_max",
        auto_discretization: bool = False,
        segment_size: int = 48,
        num_segments: int = 4
    ):
        self.__arr = arr
        self.__scaling_type = ScalingType[scaling]

        self.__min_v = np.min(arr)
        self.__max_v = np.max(arr)
        self.__mean_v = np.mean(arr)
        self.__std_v = np.std(arr)
        self.__median_v = np.median(arr)
        self.__iqr = np.percentile(arr, [75])[0] - np.percentile(arr, [25])[0]

        self.__auto_discretization = auto_discretization
        self.__segment_size = segment_size if segment_size >= 1 else 1
        self.__num_segments = num_segments if num_segments >= 1 else 1
        self.__discretizer = None

    @property
    def scaling_type(self):
        return self.__scaling_type

    @scaling_type.setter
    def scaling_type(self, scaling: str):
        self.__scaling_type = ScalingType[scaling]

    @property
    def min(self):
        return self.__min_v
    
    @property
    def max(self):
        return self.__max_v
    
    @property
    def mean(self):
        return self.__mean_v
    
    @property
    def std(self):
        return self.__std_v
    
    @property
    def iqr(self):
        return self.__iqr
    
    @property
    def median(self):
        return self.__median_v
    
    @property
    def data(self):
        return self.__arr

    @property
    def data_scaled(self):
        return self.__scaling_type.scale(self.__arr)

    @property
    def data_discrete(self):
        self.__init_discretizer()
        return self.__discretizer.discretize(self.__arr)

    def scale(
        self,
        arr: np.ndarray
    ) -> np.ndarray:
        return self.__scaling_type.scale(arr)

    def scale_reverse(
        self, 
        arr: np.ndarray
    ) -> np.ndarray:
        return self.__scaling_type.scale_reverse(
            arr = arr,
            min_v = self.__min_v,
            max_v = self.__max_v,
            mean_v = self.__mean_v,
            std_v = self.__std_v,
            median_v = self.__median_v,
            iqr = self.__iqr
        )

    def discretize(
        self, 
        arr: np.ndarray
    ) -> np.ndarray:
        self.__init_discretizer()
        return self.__discretizer.discretize(arr)

    def __init_discretizer(self):
        if self.__discretizer is None:
            self.__discretizer = Discretizer(
                self.__arr,
                auto = self.__auto_discretization,
                segment_size = self.__segment_size,
                num_segments = self.__num_segments
            )
