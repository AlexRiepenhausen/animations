import os
import cv2
import trimap
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import matplotlib.pyplot as plt

from utils import Utils


def remap_file_paths(img_paths, queue_idxs):
    file_path_dict = {}
    resorted_img_paths = np.array(img_paths)[queue_idxs]
    for i in range(len(queue_idxs)):
        original_path, new_path = img_paths[i], resorted_img_paths[i]
        file_path_dict[new_path] = original_path.replace("x.PNG", ".PNG")
    return file_path_dict


def standardize_dims(flat_images):
    flat_images = flat_images.T
    for i in range(flat_images.shape[0]):
        mean_val, stdev = np.mean(flat_images[i]), np.std(flat_images[i])
        flat_images[i] = (flat_images[i] - mean_val) / stdev
    return flat_images.T


def get_img_paths():
    img_paths = []
    for i in range(1, 323):
        in_path = f"./images/{i:04d}x.PNG"
        img_paths.append(in_path)
    return img_paths


def get_reshaped_images(img_paths, side_len=64):
    resized_images = []
    for in_path in tqdm(img_paths):
        img = cv2.imread(in_path)
        img = cv2.resize(img, (side_len, side_len), interpolation=cv2.INTER_CUBIC)
        img = cv2.filter2D(img, -1, Utils.smoothing_kernel_gauss(dim=5, nsig=3.0))
        img = img.flatten()
        resized_images.append(img.astype(np.float32))
    return np.array(resized_images)


def nearest_neighbour_random_walk(resized_images):

    path_indices = np.arange(0, len(resized_images))
    
    already_taken = set()
    queue_idxs = []
    current_img = resized_images[0]

    while len(already_taken) < len(resized_images):

        sims = cosine_similarity([current_img], resized_images)[0]
        idxs = np.argsort(sims)[::-1]
        path_indices_sorted = path_indices[idxs][1:]

        for idx in path_indices_sorted:
            if idx not in already_taken:
                queue_idxs.append(idx)
                already_taken.add(idx)
                current_img = resized_images[idx]
                break

    return np.array(queue_idxs, dtype=np.int32)


def run_main():

    img_paths = get_img_paths()
    resized_images = get_reshaped_images(img_paths, side_len=64)
    resized_images = dimensionality_reduction(resized_images, dim_size=32)
    queue_idxs = nearest_neighbour_random_walk(resized_images)
    file_path_dict = remap_file_paths(img_paths, queue_idxs)

    for original_file_path in file_path_dict:
        renamed_file_path = file_path_dict[original_file_path]
        print(original_file_path, renamed_file_path)
        os.rename(original_file_path, renamed_file_path)



def dimensionality_reduction(resized_images, dim_size=128):

    print("Compute Distance Matrix:", datetime.now())
    dist_matrix = euclidean_distances(resized_images)
    print("Finished computing distance matrix:", datetime.now())

    vectors = trimap.TRIMAP(
        n_dims=dim_size,
        use_dist_matrix=True,
        lr=0.01,
        weight_temp=0.5,
        verbose=True,
        n_iters=1200,
        apply_pca=False,
        n_inliers=24,
        n_outliers=8,
        n_random=6
    ).fit_transform(dist_matrix)

    vectors = standardize_dims(vectors)

    return vectors


if __name__ == "__main__":
    run_main()
