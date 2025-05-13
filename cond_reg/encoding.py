import numpy as np
from typing import Dict

class Encoding:

    @staticmethod
    def discretize(
        arr: np.ndarray,
        bins: np.ndarray
    ) -> np.ndarray:
        if np.isscalar(arr):
            arr = np.array([arr])
        discr = np.digitize(arr, bins = bins, right = True)
        idxs = np.where(discr == bins.shape[0])
        discr[idxs] = bins.shape[0] - 1
        return discr

    @staticmethod
    def integer_encoding(
        arr: np.ndarray,
        cat2int: Dict
    ) -> np.ndarray:
        return np.array([cat2int[c] for c in arr]).astype(np.int32)

    @staticmethod
    def integer_decoding(
        arr: np.ndarray,
        int2cat: Dict     
    ) -> np.ndarray:
        return np.array([int2cat[int(item)] for item in arr]).astype(np.int32)

    @staticmethod
    def get_cont_var_segments_to_idxs(
        arr: np.ndarray, 
        segment_size: int = 48
    ) -> Dict:

        idxs_sort = np.argsort(arr).astype(np.int32)
        arr_sort = arr[idxs_sort]

        arr_sort_diff = np.diff(arr_sort)
        borders = np.where(arr_sort_diff > 0.0)[0] + 1

        seg_id, s_start, s_end = 0, 0, 0
        num_brdrs, num_items = borders.shape[0], arr.shape[0]

        segment_ids, segment_idxs = [], []

        if num_items > segment_size:
            for brdr_id in range(num_brdrs):
                s_end = borders[brdr_id]
                if s_end - s_start >= segment_size:
                    segment_ids.append(seg_id)
                    segment_idxs.append(idxs_sort[s_start:s_end])
                    s_start = s_end
                    seg_id += 1

            if num_items - s_end > 0:
                if num_items - s_end >= segment_size // 2:
                    segment_ids.append(seg_id)
                    segment_idxs.append(idxs_sort[s_start:])
                else:
                    if len(segment_ids) > 0 and len(idxs_sort[s_start:]) > 0:
                        current_seg_id = int(np.max(segment_ids))
                        segment_idxs[current_seg_id] = np.concatenate((segment_idxs[current_seg_id], idxs_sort[s_start:]))
        else:
            segment_ids = [0]
            segment_idxs = [idxs_sort]

        seg2idxs = {seg_id: segment_idxs[seg_id].astype(np.int32) for seg_id in segment_ids}

        return seg2idxs
    
    @staticmethod
    def get_cat_var_segments_to_idxs(arr: np.ndarray) -> Dict:
        seg2idxs = {cat: [] for cat in sorted(np.unique(arr))}
        for i, cat in enumerate(arr):
            seg2idxs[cat].append(i)
        for cat, idxs in seg2idxs.items():
            seg2idxs[cat] = np.array(idxs).astype(np.int32)
        return seg2idxs

    @staticmethod
    def get_bins_from_segments_to_idxs(
        arr: np.ndarray,
        seg2idxs: Dict
    ) -> np.ndarray:
        bins = []
        segment_ids = sorted(list(seg2idxs.keys()))
        for seg_id in segment_ids:
            idxs = seg2idxs[seg_id]
            bins.append(np.max(arr[idxs]))
        bins = np.array(bins).astype(np.float32)
        return bins

    @staticmethod
    def get_bins_from_range(arr: np.ndarray, num_segments: int) -> np.ndarray:
        min_val, max_val = np.min(arr), np.max(arr)
        rng = (max_val - min_val) / num_segments
        return np.arange(min_val + rng, max_val + rng, rng)

    @staticmethod
    def get_int2cat_dict(cat2int: Dict) -> Dict:
        return {int(i): cat for cat, i in cat2int.items()}

    @staticmethod
    def get_cat2int_dict(arr: np.ndarray) -> Dict:
        if arr.shape[0] > 0:
            return {item: int(i) for i, item in enumerate(np.unique(arr))}
        else:
            return {}
