import cv2
import numpy as np
from glob import glob
import os


def mse_calc(truth_files, results_templ):
    total_mse = 0.

    for truth_file in truth_files:
        base = os.path.splitext(os.path.basename(truth_file))[0]
        output_file = results_templ % base

        img_truth = cv2.imread(truth_file).astype(np.float32) / 255.
        img_output = cv2.imread(output_file).astype(np.float32) / 255.

        if img_truth.shape != img_output.shape:
            img_output = cv2.resize(img_output, (img_truth.shape[1], img_truth.shape[0]), interpolation=cv2.INTER_CUBIC)

        mse = np.sum((img_truth - img_output) ** 2) / (img_truth.shape[0] * img_truth.shape[1])
        total_mse += mse

    return total_mse


truth_files = sorted(glob('./datasets/noisytext/testB/*.png'))
results_dirs = sorted(glob('./results/noisytext_cyclegan/*'))


def gan_mse_search():
    mse_min = np.inf
    best_results = ""

    for results_dir in results_dirs:
        results_templ = os.path.join(results_dir, 'images/%s_fake_B.png')
        total_mse = mse_calc(truth_files, results_templ)

        if total_mse < mse_min:
            mse_min = total_mse
            best_results = results_dir

        print(results_dir, total_mse, best_results, mse_min)

    print()
    print("=== BEST GAN RESULT ===")
    print(best_results, mse_min)
    print()


cyclegan_mse = mse_calc(truth_files, "./results/noisytext_cyclegan/test_15/images/%s_fake_B.png")
print("=== CYCLEGAN RESULT ===")
print(cyclegan_mse)

stacker_mse = mse_calc(truth_files, "./results/stacker_test_outputs_full/stacker_cleaned_%s.png")
print("=== STACKER RESULT ===")
print(stacker_mse)

autoencoder_mse = mse_calc(truth_files, "./results/autoencoder_test_outputs_full/autoencoder_cleaned_%s.png")
print("=== AUTOENCODER RESULT ===")
print(autoencoder_mse)

adaptivefiltered_mse = mse_calc(truth_files, "./results/adaptive_filtered/%s.png")
print("=== ADAPTIVEFILTERED RESULT ===")
print(adaptivefiltered_mse)

edgedilationerosion_mse = mse_calc(truth_files, "./results/edge_dilation_erosion/%s.png")
print("=== EDGEDILATIONEROSION RESULT ===")
print(edgedilationerosion_mse)

medianfiltered_mse = mse_calc(truth_files, "./results/median_filtered/%s.png")
print("=== MEDIANFILTERED RESULT ===")
print(medianfiltered_mse)
